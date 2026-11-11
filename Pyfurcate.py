"""
Whole-brain model of Stuart-Landau oscillators at Hopf bifurcation

https://doi.org/10.1371/journal.pcbi.1000196
https://doi.org/10.1371/journal.pcbi.1002634
https://doi.org/10.1038/s41598-017-03073-5

Perturbations:
https://doi.org/10.1371/journal.pcbi.1010662
Delays:
https://doi.org/10.1038/s42005-022-00950-y
Generative Effective Connectivity (GEC):
https://doi.org/10.1073/pnas.1905534116
Competitive interactions:
https://doi.org/10.1101/2024.10.19.619194

Authors: George Blackburne, Hanna Tolle
20/05/25
"""

import time
import numpy as np
import warnings
import networkx as nx
from scipy import signal
import matplotlib.pyplot as plt

from numba import jit, float64, int64
from numba.core.errors import NumbaPerformanceWarning
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

import torch
import torch.nn as nn
import torch.optim as optim

torch.backends.cudnn.benchmark = True

if hasattr(torch, 'compile'):
    torch.set_float32_matmul_precision('high')


def _example_params(nnodes=25):
    """Example model parameters for a small-world network."""
    return {
        "nnodes":       nnodes,
        "C":            np.array(nx.to_numpy_array(
                            nx.watts_strogatz_graph(nnodes, 5, 0.1, 0))) / 10,
        "G":            1,
        "dt":           1,
        "teq":          60,
        "tmax":         1200,
        "downsamp":     1,
        "a":            np.zeros((nnodes, 1)),
        "w":            0.05 * np.ones((nnodes, 1)),
        "F0":           np.zeros((nnodes, 1)),
        "F0_w":         0.05 * np.ones((nnodes, 1)),
        "beta":         1e-2,
        "sigma":        0,
        "use_filt":     False,
        "low_cut":      0.008,
        "high_cut":     0.09,
        "filter_order": 2,
    }


def fmri_params(TR, duration, teq=None, high_cut=0.09, max_dt=1, verbose=True):
    """
    Convert fMRI acquisition parameters into HopfModel timing parameters.
    TR is subdivided into steps of at most max_dt for numerical stability,
    with downsamp set so that output spacing matches TR exactly.

    Parameters
    ----------
    TR : float
        Repetition time (s) — temporal resolution of the fMRI data.
    duration : float
        Total scan duration (s). Sets tmax.
    teq : float, optional
        Equilibration (burn-in) time (s). Defaults to max(60, duration * 0.1).
    high_cut : float, optional
        High cutoff of bandpass filter (Hz). Used only to check TR is short
        enough to satisfy the Nyquist constraint. Default 0.09 Hz.
    max_dt : float, optional
        Maximum integration timestep (s). TR is subdivided into
        ceil(TR / max_dt) steps. Default 1 s.

    Returns
    -------
    params : dict
        {"dt", "downsamp", "tmax", "teq"} — pass directly to HopfModel.

    Examples
    --------
    >> model = HopfModel(nnodes=232, C=SC, w=w, **fmri_params(TR=2.0, duration=600))
    """
    nyquist = 0.5 / TR
    if high_cut >= nyquist:
        raise ValueError(
            f"TR={TR}s gives Nyquist={nyquist:.4f} Hz, which is at or below "
            f"high_cut={high_cut} Hz. Use TR < {1.0/(2.0*high_cut):.2f}s, or lower high_cut."
        )

    n = int(np.ceil(TR / max_dt))
    dt = TR / n
    downsamp = n

    if teq is None:
        teq = max(60.0, duration * 0.1)

    if verbose:
        print(f"fMRI parameters: TR={TR}s, duration={duration}s")
        print(f" --------------> dt={dt:.3f}s, tmax={duration:.1f}s, "
              f"downsamp={downsamp}, teq={teq:.1f}s")

    return {
        "dt":       dt,
        "downsamp": downsamp,
        "tmax":     float(duration),
        "teq":      float(teq),
    }


@jit(float64[:,:](float64[:,:], float64[:,:], float64, float64[:,:],
                  float64, float64[:,:], float64[:,:], float64[:,:], float64[:,:]),
     nopython=True)
def hopf_dynamics_numba(x, y, t, C, G, a, w, F0, F0_w):
    """
    Euler-Maruyama step for a network of Stuart-Landau oscillators (no delay).
    """
    nnodes = x.shape[0]
    ones_vector = np.ones(nnodes).reshape((1, nnodes))

    # single matrix vector multiply for memory saving, avoids the NxN diference matrix
    row_sum = C @ ones_vector.T
    IsynX = G * (C @ x - row_sum * x)
    IsynY = G * (C @ y - row_sum * y)

    x_dot = (a - x**2 - y**2) * x - w * y + IsynX + F0 * np.cos(t * F0_w)
    y_dot = (a - x**2 - y**2) * y + w * x + IsynY + F0 * np.sin(t * F0_w)

    return np.hstack((x_dot, y_dot))


@jit(float64[:,:](float64, float64, float64[:,:], int64, float64), nopython=True)
def noise_numba(beta, sigma, F0, nnodes, dt):
    """
    Isotropic white noise for single step.
    """
    scale = (beta + sigma * F0) * np.sqrt(dt)
    x_dot = np.random.normal(0, 1, (nnodes, 1)) * scale
    y_dot = np.random.normal(0, 1, (nnodes, 1)) * scale
    return np.hstack((x_dot, y_dot))


def bandpass_numpy(data, sfreq,
                   low_cut=0.008, high_cut=0.09,
                   filter_order=2):
    """
    Butterworth bandpass filter (zero-phase, SOS form) via SciPy.

    Parameters
    ----------
    data : np.ndarray  (time x nodes)
    sfreq : float      Sampling frequency (Hz)
    low_cut, high_cut : float  Cut-off frequencies (Hz)
    filter_order : int  Butterworth order

    Returns
    -------
    filtered_data : np.ndarray  (time x nodes)
    """
    nyquist = 0.5 * sfreq
    low  = low_cut  / nyquist
    high = high_cut / nyquist
    # SOS form is numerically stable at very low normalised frequencies
    sos = signal.butter(filter_order, [low, high], btype='band', output='sos')
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        filtered_data[:, i] = signal.sosfiltfilt(sos, data[:, i])
    return filtered_data

def bandpass_torch(data, sfreq, low_cut=0.008, high_cut=0.09, filter_order=2):
    """
    Butterworth bandpass IIR filter in PyTorch, matching scipy.signal.sosfiltfilt.

    Filter coefficients and initial conditions are computed once via scipy (not
    tracked by autograd). The IIR recursion is pure PyTorch and is fully
    differentiable.

    Parameters
    ----------
    data : torch.Tensor  (time x nodes)
    sfreq : float        Sampling frequency (Hz)
    low_cut, high_cut : float  Cut-off frequencies (Hz)
    filter_order : int   Butterworth order

    Returns
    -------
    filtered : torch.Tensor  (time x nodes)
    """
    device = data.device
    dtype  = data.dtype

    nyq    = 0.5 * sfreq
    sos_np = signal.butter(filter_order, [low_cut / nyq, high_cut / nyq],
                           btype='band', output='sos')
    zi_np  = signal.sosfilt_zi(sos_np)   # (n_sections, 2) per-unit steady-state IC

    sos_t = torch.tensor(sos_np, dtype=dtype, device=device)
    zi_t  = torch.tensor(zi_np,  dtype=dtype, device=device)

    x = data.T.contiguous()          # (nodes, time)
    n_sections = sos_np.shape[0]

    npad = 3 * (2 * n_sections + 1)
    xl = 2.0 * x[:, :1]  - x[:,  1:npad + 1].flip(dims=[1])
    xr = 2.0 * x[:, -1:] - x[:, -(npad + 1):-1].flip(dims=[1])
    xp = torch.cat([xl, x, xr], dim=1)     # (nodes, time + 2*npad)

    def _sosfilt(sig, scale):
        y = sig
        for s in range(n_sections):
            b0, b1, b2, a0, a1, a2 = sos_t[s]
            z0 = zi_t[s, 0] * scale
            z1 = zi_t[s, 1] * scale
            outs = []
            for t in range(y.shape[1]):
                yt = (b0 * y[:, t] + z0) / a0
                z0, z1 = (b1 * y[:, t] - a1 * yt + z1,
                          b2 * y[:, t] - a2 * yt)
                outs.append(yt)
            y = torch.stack(outs, dim=1)
        return y

    yp = _sosfilt(xp, xp[:, 0])           # forward pass
    yp = yp.flip(dims=[1])
    yp = _sosfilt(yp, yp[:, 0])           # reverse pass (filtfilt)
    yp = yp.flip(dims=[1])

    return yp[:, npad:-npad].T



class HopfModel(nn.Module):
    """
    Whole-brain model of Stuart-Landau oscillators.

    Dual-backend
    -------------------
    backend='numpy'  Fast NumPy / Numba simulation for exploration and
                     analytical-gradient GEC optimisation. Supports both
                     conduction delays and time-gated stimulation.
    backend='torch'  Differentiable PyTorch simulation for autograd GEC
                     optimisation. Supports time-gated stimulation but not
                     conduction delays.
    """

    def __init__(self,
                 nnodes, C,
                 backend='numpy',
                 G=1,
                 a=0, w=0.05,
                 F0=0, F0_w=0.05,
                 beta=1e-2, sigma=0,
                 dt=1, teq=60, tmax=1200, downsamp=1,
                 use_filt=False, low_cut=0.008, high_cut=0.09, filter_order=2,
                 tbptt_k=None,
                 stim_duration=None, stim_onset=0.0,
                 D=None, MD=0.0,
                 device='cpu'):
        """
        Parameters
        ----------
        nnodes : int
            Number of nodes in the network.
        C : np.ndarray  (nnodes x nnodes)
            Structural/effective connectivity matrix. Pass None (torch only)
            to initialise GEC with small random weights.
        backend : str
            'numpy'  — fast NumPy/Numba simulation (supports delays + stim gating).
            'torch'  — differentiable PyTorch simulation (supports stim gating only).
        G : float
            Global coupling scaling factor.
        a : float or np.ndarray  (nnodes, 1)
            Bifurcation parameter per node. a < 0 → damped; a > 0 → limit cycle.
        w : float or np.ndarray  (nnodes, 1)
            Natural oscillation frequency per node (Hz). Converted to rad/s internally.
        F0 : float or np.ndarray  (nnodes, 1)
            Amplitude of external sinusoidal forcing per node.
        F0_w : float or np.ndarray  (nnodes, 1)
            Frequency of external forcing per node (Hz). Converted to rad/s internally.
        beta : float
            Isotropic noise amplitude. Each of x and y receives independent
            N(0, beta^2 * dt) noise, giving total complex noise variance
            2 * beta^2 * dt per node.
        sigma : float
            Additional noise for stimulated nodes. Per-component noise amplitude
            becomes (beta + sigma * F0), so forcing nodes receive more noise.
        dt : float
            Euler-Maruyama integration timestep (s).
        teq : float
            Equilibration / burn-in duration (s) discarded before recording.
        tmax : float
            Recorded simulation duration (s).
        downsamp : int
            Store one output sample every `downsamp` integration steps.
        use_filt : bool
            Apply Butterworth bandpass filter to x before returning.
        low_cut, high_cut : float
            Bandpass filter cut-off frequencies (Hz).
        filter_order : int
            Butterworth filter order.
        tbptt_k : int or None
            (Torch backend) Steps from the end through which gradients flow.
            None → full BPTT (expensive for long tmax).
        stim_duration : float or None
            Duration of the stimulation window (s).
            None (default) - stimulation constant
            (fully backward compatible with the original model).
        stim_onset : float
            Simulation time (s) at which stimulation begins. Default 0
            (from the very first integration step, including teq).
            Set to teq to start stimulation only in the recorded phase.
        D : np.ndarray or None  (nnodes x nnodes)
            Fibre-length distance matrix (any units consistent with MD).
            Numpy backend only. Default None (no delays).
        MD : float
            Mean conduction delay (s). D is rescaled so that the mean delay
            across connected pairs equals MD. Numpy backend only. Default 0.
            Note: meaningful delays require dt << MD.
        device : str
            PyTorch device ('cpu' or 'cuda'). Ignored for numpy backend.
        """

        super(HopfModel, self).__init__()

        self.backend      = backend
        self.nnodes       = nnodes
        self.dt           = dt
        self.teq          = teq
        self.tmax         = tmax
        self.downsamp     = downsamp
        self.use_filt     = use_filt
        self.low_cut      = low_cut
        self.high_cut     = high_cut
        self.filter_order = filter_order

        self.stim_duration = stim_duration
        self.stim_onset    = float(stim_onset)

        # Broadcast scalar parameters to per-node arrays
        if isinstance(a,    (int, float)): a    = a    * np.ones((nnodes, 1))
        if isinstance(w,    (int, float)): w    = w    * np.ones((nnodes, 1))
        if isinstance(F0,   (int, float)): F0   = F0   * np.ones((nnodes, 1))
        if isinstance(F0_w, (int, float)): F0_w = F0_w * np.ones((nnodes, 1))

        # Numerical stability warnings
        w_max_rads = float(np.max(w)) * 2 * np.pi
        if w_max_rads > 0 and dt >= 2.0 / w_max_rads:
            warnings.warn(
                f"dt={dt} may be too large for stable integration "
                f"(max w={np.max(w):.4f} Hz -> stability limit dt < "
                f"{2.0/w_max_rads:.2f}s). Consider reducing dt.",
                RuntimeWarning, stacklevel=2
            )
        min_freq_hz  = float(np.min(w))
        n_cycles     = min_freq_hz * tmax
        n_timepoints = tmax / dt / downsamp
        if min_freq_hz > 0 and n_cycles < 20 and n_timepoints >= 200:
            warnings.warn(
                f"Slowest oscillator ({min_freq_hz:.4f} Hz) completes only "
                f"{n_cycles:.1f} cycles over tmax={tmax}s. FC estimates may be "
                f"unreliable. Consider tmax >= {20 / min_freq_hz:.0f}s.",
                RuntimeWarning, stacklevel=2
            )

        # Convert Hz to rad/s for internal use
        w    = w    * 2 * np.pi
        F0_w = F0_w * 2 * np.pi

        a    = a.reshape(nnodes, 1)    if a.ndim    == 1 else a
        w    = w.reshape(nnodes, 1)    if w.ndim    == 1 else w
        F0   = F0.reshape(nnodes, 1)   if F0.ndim   == 1 else F0
        F0_w = F0_w.reshape(nnodes, 1) if F0_w.ndim == 1 else F0_w

        if backend == 'torch':

            self.device = torch.device(device)

            C_np   = np.zeros((nnodes, nnodes)) if C is None else C
            self.C       = torch.tensor(C_np,  dtype=torch.float32, device=self.device)
            self.G       = torch.tensor(G,     dtype=torch.float32, device=self.device)
            self.dt_tensor = torch.tensor(dt,  dtype=torch.float32, device=self.device)
            self.a       = torch.tensor(a,     dtype=torch.float32, device=self.device)
            self.w       = torch.tensor(w,     dtype=torch.float32, device=self.device)
            self.F0      = torch.tensor(F0,    dtype=torch.float32, device=self.device)
            self.F0_w    = torch.tensor(F0_w,  dtype=torch.float32, device=self.device)
            self.beta    = torch.tensor(beta,  dtype=torch.float32, device=self.device)
            self.sigma   = torch.tensor(sigma, dtype=torch.float32, device=self.device)
            self.ones_vector = torch.ones(1, self.nnodes, dtype=torch.float32,
                                          device=self.device)

            # Stim gating: zero-F0 tensor and a per-step pointer updated in the loop
            self._F0_zero     = torch.zeros_like(self.F0)
            self._current_F0  = self.F0   # updated each step in _forward_torch

            # GEC: learned connectivity (initialised from C or random)
            if C is None:
                gec_init = torch.empty(nnodes, nnodes, dtype=torch.float32,
                                       device=self.device).uniform_(-0.005, 0.005)
                gec_init.fill_diagonal_(0)
                self.GEC = nn.Parameter(gec_init)
            else:
                self.GEC = nn.Parameter(
                    torch.tensor(C_np, dtype=torch.float32, device=self.device))

            # Truncated BPTT window (None = full BPTT)
            self.tbptt_k  = tbptt_k
            # Delays not implemented for the torch backend
            self.max_delay = 0

            if hasattr(torch, 'compile') and torch.cuda.is_available():
                self.hopf_dynamics_torch = torch.compile(self.hopf_dynamics_torch)

        else:  # numpy backend
            if C is None:
                raise ValueError("C must be provided for the numpy backend.")
            self.C     = C.astype(np.float64)
            self.G     = np.float64(G)
            self.a     = a.astype(np.float64)
            self.w     = w.astype(np.float64)
            self.F0    = F0.astype(np.float64)
            self.F0_w  = F0_w.astype(np.float64)
            self.beta  = np.float64(beta)
            self.sigma = np.float64(sigma)
            self.GEC   = np.copy(self.C)

            # Stim gating: zero arrays used when stimulation is off
            self._F0_zero    = np.zeros_like(self.F0)
            self._sigma_zero = np.float64(0.0)

            # Delays
            if D is not None and MD > 0:
                D_arr = np.array(D, dtype=np.float64)
                connected = self.C > 0
                mean_D = D_arr[connected].mean() if connected.any() else 1.0
                Delays = np.round(D_arr / mean_D * MD / dt).astype(np.int64) # rescale and convert to steps
                Delays[~connected] = 0
                self.max_delay = int(Delays.max())
                if self.max_delay == 0:
                    warnings.warn(
                        f"MD={MD}s with dt={dt}s: all delays round to 0 steps. "
                        f"Increase MD or decrease dt for meaningful delays.",
                        RuntimeWarning, stacklevel=2
                    )
            else:
                Delays = np.zeros((nnodes, nnodes), dtype=np.int64)
                self.max_delay = 0

            # delay_idx[n,p]: column index into the (nnodes × H) history buffer
            # for the state of node p as seen by node n.
            self.delay_idx = self.max_delay - Delays

    def _stim_active(self, t):
        """ Return True if stim should be applied at time t (s). """
        if self.stim_duration is None:
            return True
        return self.stim_onset <= t < self.stim_onset + self.stim_duration
    
    def hopf_dynamics_torch(self, x, y, t):
        # single matrix vector multiply for memory saving, avoids the NxN diference matrix
        row_sum = self.GEC.sum(1, keepdim=True)
        IsynX = self.G * (self.GEC @ x - row_sum * x)
        IsynY = self.G * (self.GEC @ y - row_sum * y)

        F0 = self._current_F0
        x_dot = (self.a - x**2 - y**2) * x - self.w * y + IsynX + F0 * torch.cos(t * self.F0_w)
        y_dot = (self.a - x**2 - y**2) * y + self.w * x + IsynY + F0 * torch.sin(t * self.F0_w)

        return x_dot, y_dot
    
    def forward(self, mask=None, allow_negative=True, verbose=False):
        """
        Run the Hopf model simulation.

        Parameters
        ----------
        mask : array-like, optional
            Binary mask of connections to update (torch backend only).
        allow_negative : bool
            Allow negative GEC weights (torch backend only).
        verbose : bool
            Print elapsed time every 10 s of simulated time (numpy backend).

        Returns
        -------
        results : ndarray or Tensor  (time x nnodes x 2)
            Trajectories for x (index 0) and y (index 1) state variables.
        time_vector : ndarray or Tensor  (time,)
            Time points for the recorded portion of the simulation.
        """
        if self.backend == 'torch':
            return self._forward_torch(mask, allow_negative)
        else:
            return self._forward_numpy(verbose)

    def _forward_torch(self, mask=None, allow_negative=True):

        # Enforce connectivity constraints on GEC before integration
        if mask is not None:
            self.GEC.data *= mask
        if not allow_negative:
            self.GEC.data.clamp_(min=0)

        Neq  = int(self.teq  / self.dt / self.downsamp)
        Nmax = int(self.tmax / self.dt / self.downsamp)
        Nsim = (Nmax + Neq - 1) * self.downsamp + 1

        # Random initial conditions in (0, 1)
        x = torch.ones(self.nnodes, 1, device=self.device) * \
            (torch.rand(self.nnodes, 1, device=self.device) * 0.99 + 0.01)
        y = torch.ones(self.nnodes, 1, device=self.device) * \
            (torch.rand(self.nnodes, 1, device=self.device) * 0.99 + 0.01)

        # Pre-sample noise; scale is time-gated when stim_duration is not None
        sqrt_dt = torch.sqrt(self.dt_tensor)
        if self.stim_duration is None:
            # Always-on path: single uniform noise scale
            noise_scale = self.beta + self.sigma * self.F0
            all_noise_x = (torch.randn(Nsim, self.nnodes, 1, device=self.device)
                           * noise_scale * sqrt_dt)
            all_noise_y = (torch.randn(Nsim, self.nnodes, 1, device=self.device)
                           * noise_scale * sqrt_dt)
        else:
            # Time-gated path: per-step stim flags modulate the noise scale
            noise_scale_stim = self.beta + self.sigma * self.F0
            noise_scale_no_stim = self.beta + torch.zeros_like(self.F0)
            stim_flags = torch.tensor(
                [1.0 if self._stim_active(i * self.dt) else 0.0
                 for i in range(Nsim)],
                dtype=self.F0.dtype, device=self.device
            ).view(Nsim, 1, 1)
            noise_scale = (noise_scale_no_stim
                           + stim_flags * (noise_scale_stim - noise_scale_no_stim))
            all_noise_x = (torch.randn(Nsim, self.nnodes, 1, device=self.device)
                           * noise_scale * sqrt_dt)
            all_noise_y = (torch.randn(Nsim, self.nnodes, 1, device=self.device)
                           * noise_scale * sqrt_dt)

        x_states = [x.squeeze()]
        y_states = [y.squeeze()]

        for i in range(1, Nsim):
            # Gate the forcing amplitude for this step
            self._current_F0 = (self.F0 if self._stim_active(i * self.dt)
                                else self._F0_zero)

            x_dot, y_dot = self.hopf_dynamics_torch(x, y, i * self.dt_tensor)
            x = x + x_dot * self.dt_tensor + all_noise_x[i]
            y = y + y_dot * self.dt_tensor + all_noise_y[i]

            # Truncated BPTT: detach states outside the gradient window
            if self.tbptt_k is not None and i < Nsim - self.tbptt_k:
                x = x.detach()
                y = y.detach()

            if (i % self.downsamp) == 0:
                x_states.append(x.squeeze())
                y_states.append(y.squeeze())

        results_x = torch.stack(x_states, dim=0)   # (Nmax+Neq, nnodes)
        results_y = torch.stack(y_states, dim=0)
        results = torch.stack([results_x, results_y], dim=2)

        # Discard equilibration period
        results = results[Neq:]

        if self.use_filt:
            sfreq = 1 / (self.dt * self.downsamp)
            nyquist = 0.5 * sfreq
            if self.high_cut >= nyquist:
                raise ValueError(
                    f"high_cut ({self.high_cut} Hz) must be < Nyquist "
                    f"({nyquist:.4f} Hz). Reduce high_cut or downsamp.")
            if self.low_cut <= 0 or self.low_cut >= nyquist:
                raise ValueError(
                    f"low_cut ({self.low_cut} Hz) must be in (0, {nyquist:.4f} Hz).")
            filtered_x = bandpass_torch(
                results[:, :, 0], sfreq, self.low_cut, self.high_cut, self.filter_order)
            results = torch.stack([filtered_x, results[:, :, 1]], dim=2)

        time_vector = torch.linspace(0, self.tmax, Nmax, device=self.device)
        return results, time_vector

    def _forward_numpy(self, verbose=False):

        Neq = int(self.teq  / self.dt / self.downsamp)
        Nmax = int(self.tmax / self.dt / self.downsamp)
        Nsim = (Nmax + Neq - 1) * self.downsamp + 1

        results = np.zeros((Nmax + Neq, self.nnodes, 2))

        if self.max_delay > 0: # No JIT compilation for buffer
            # x_hist[:, -1] is the current state; x_hist[:, 0] is max_delay steps ago.
            H = self.max_delay + 1
            x_hist = np.random.uniform(0.01, 1, (self.nnodes, H))
            y_hist = np.random.uniform(0.01, 1, (self.nnodes, H))
            results[0, :, 0] = x_hist[:, -1]
            results[0, :, 1] = y_hist[:, -1]

            p_idx = np.broadcast_to(
                np.arange(self.nnodes)[None, :], (self.nnodes, self.nnodes))

            # Row sums of GEC: sum_p(C_ip) for each node i
            gec_row_sum = self.GEC.sum(axis=1, keepdims=True)
            # Pre-compute noise scales for stim-on and stim-off states
            # shape: (nnodes, 1) — allows per-node noise amplitude variation
            noise_scale_on  = (self.beta + self.sigma * self.F0) * np.sqrt(self.dt)
            noise_scale_off = np.ones((self.nnodes, 1)) * self.beta * np.sqrt(self.dt)

            for i in range(1, Nsim):
                t = i * self.dt

                # Determine stim state and select F0 / noise scale accordingly
                stim_on = self._stim_active(t)
                F0_now = self.F0 if stim_on else self._F0_zero
                noise_scale = noise_scale_on if stim_on else noise_scale_off

                # Current state of all nodes
                x = x_hist[:, [-1]]   # (nnodes, 1)
                y = y_hist[:, [-1]]
                # Delayed states: x_del[n, p] = state of p seen by n (with delay)
                x_del = x_hist[p_idx, self.delay_idx]   # (nnodes, nnodes)
                y_del = y_hist[p_idx, self.delay_idx]

                # Diffusive coupling with delays:
                IsynX = self.G * (
                    (self.GEC * x_del).sum(1, keepdims=True) - gec_row_sum * x)
                IsynY = self.G * (
                    (self.GEC * y_del).sum(1, keepdims=True) - gec_row_sum * y)

                # Hopf intrinsic dynamics
                r2 = x**2 + y**2
                x_dot = ((self.a - r2) * x - self.w * y + IsynX
                         + F0_now * np.cos(t * self.F0_w))
                y_dot = ((self.a - r2) * y + self.w * x + IsynY
                         + F0_now * np.sin(t * self.F0_w))

                # Integration with time-gated noise scale
                xi_x = np.random.normal(0, 1, (self.nnodes, 1))
                xi_y = np.random.normal(0, 1, (self.nnodes, 1))
                x_new = x + x_dot * self.dt + xi_x * noise_scale
                y_new = y + y_dot * self.dt + xi_y * noise_scale

                x_hist[:, :-1] = x_hist[:, 1:] # shift history buffer
                y_hist[:, :-1] = y_hist[:, 1:]
                x_hist[:, -1]  = x_new[:, 0]
                y_hist[:, -1]  = y_new[:, 0]

                if (i % self.downsamp) == 0:
                    results[i // self.downsamp, :, 0] = x_new[:, 0]
                    results[i // self.downsamp, :, 1] = y_new[:, 0]

                if verbose and self.dt <= 10 and (i % int(10 / self.dt)) == 0:
                    print(f'Elapsed time: {t:.0f} s')

        else: # fast numba
            ic = np.random.uniform(0.01, 1, (self.nnodes, 2))
            results[0] = np.copy(ic)
            results_temp = np.copy(ic)

            for i in range(1, Nsim):
                t = i * self.dt

                if self._stim_active(t):
                    F0_now = self.F0
                    sigma_now = self.sigma
                else:
                    F0_now = self._F0_zero
                    sigma_now = self._sigma_zero

                dynamics = hopf_dynamics_numba(
                    results_temp[:, [0]], results_temp[:, [1]],
                    t, self.GEC, self.G, self.a, self.w, F0_now, self.F0_w
                )
                noise = noise_numba(
                    self.beta, sigma_now, F0_now, self.nnodes, self.dt)
                results_temp += dynamics * self.dt + noise

                if (i % self.downsamp) == 0:
                    results[i // self.downsamp] = results_temp

                if verbose and self.dt <= 10 and (i % int(10 / self.dt)) == 0:
                    print(f'Elapsed time: {t:.0f} s')

        # Discard equilibration period
        results = results[Neq:]

        if self.use_filt:
            sfreq = 1 / (self.dt * self.downsamp)
            nyquist = 0.5 * sfreq
            if self.high_cut >= nyquist:
                raise ValueError(
                    f"high_cut ({self.high_cut} Hz) must be < Nyquist "
                    f"({nyquist:.4f} Hz). Reduce high_cut or downsamp.")
            if self.low_cut <= 0 or self.low_cut >= nyquist:
                raise ValueError(
                    f"low_cut ({self.low_cut} Hz) must be in (0, {nyquist:.4f} Hz).")
            min_len = 3 * (2 * self.filter_order + 1)
            if Nmax <= min_len:
                raise ValueError(
                    f"Simulated data ({Nmax} samples) too short for filtfilt "
                    f"(needs > {min_len}). Increase tmax or reduce dt/downsamp.")
            results[:, :, 0] = bandpass_numpy(
                results[:, :, 0], sfreq, self.low_cut, self.high_cut, self.filter_order)

        time_vector = np.linspace(0, self.tmax, Nmax)
        return results, time_vector

    def get_GEC(self):
        """Return the current GEC matrix as a NumPy array."""
        if self.backend == 'torch':
            return self.GEC.detach().cpu().numpy()
        else:
            return self.GEC.copy()

    def to_BOLD(self, results):
        """
        Extract the x-component as a BOLD-like signal.

        Parameters
        ----------
        results : ndarray or Tensor  (time x nnodes x 2)

        Returns
        -------
        BOLD : ndarray  (nnodes x time)
        """
        if self.backend == 'torch' and torch.is_tensor(results):
            return results[:, :, 0].detach().cpu().numpy().T
        else:
            return results[:, :, 0].T


def optimize_GEC(empirical_FC, C=None,
                 use_sc_mask=True,
                 use_analytical_grad=False,
                 optimizer_type='adam',
                 n_iterations=1000, n_simulations=2,
                 lr=2e-3, min_lr=5e-5, grad_clip=1.0,
                 scheduler_patience=80,
                 converg_tol=1e-10,
                 weight_decay=1e-5,
                 momentum=0.9,
                 max_gec=0.5,
                 gec_init_max=0.1,
                 allow_negative=True,
                 verbose=False,
                 G=1,
                 a=0, w=0.05,
                 F0=0, F0_w=0.05,
                 dt=1, teq=60, tmax=1200, downsamp=1,
                 dt_opt=None, teq_opt=60.0, tmax_opt=None, downsamp_opt=None,
                 beta=1e-2, sigma=0,
                 use_filt=False, low_cut=0.008, high_cut=0.09, filter_order=2,
                 tbptt_k=None,
                 tbptt_frac=0.4,
                 full_output=False,
                 device='cpu'):
    """
    Fit a Generative Effective Connectivity (GEC) matrix to empirical FC by
    gradient descent on the Hopf whole-brain model.

    The loss is mean-squared error in Pearson-correlation space:
        L = mean( (sim_corr[i,j] - FC_emp[i,j])^2 )  for i < j

    Suggested settings (autograd, preferred)
    -----------------------------------------
    ::

        timings = fmri_params(TR=0.72, duration=scan_duration)
        results = optimize_GEC(
            FC_emp, SC,
            use_analytical_grad = False,
            optimizer_type      = 'adam',
            lr                  = 2e-3,
            min_lr              = 5e-5,
            grad_clip           = 1.0,
            tbptt_k             = None,
            tbptt_frac          = 0.4,
            teq_opt             = 60.0,
            scheduler_patience  = 100,
            n_iterations        = 500,
            n_simulations       = 2,
            tmax_opt            = 1440,
            max_gec             = 0.5,
            gec_init_max        = 0.1,
            allow_negative      = True,
            weight_decay        = 1e-5,
            use_sc_mask         = True,
            verbose             = True,
            **timings,
        )

    Analytical delta-rule (faster on CPU, no autograd)
    ---------------------------------------------------
    ::

        timings = fmri_params(TR=0.72, duration=scan_duration)
        results = optimize_GEC(
            FC_emp, SC,
            use_analytical_grad = True,
            optimizer_type      = 'adam',
            lr                  = 3e-3,
            min_lr              = 5e-5,
            grad_clip           = 1.0,
            scheduler_patience  = 20,
            n_iterations        = 500,
            n_simulations       = 2,
            tmax_opt            = 1000,
            max_gec             = 0.5,
            gec_init_max        = 0.1,
            allow_negative      = True,
            weight_decay        = 1e-5,
            **timings,
        )

    Parameters
    ----------
    empirical_FC : np.ndarray  (nnodes x nnodes)
        Empirical FC matrix (Pearson correlations).
    C : np.ndarray or None
        Structural connectivity. None → random GEC initialisation.
    use_sc_mask : bool
        Only optimise entries where C > 0 (use SC as sparsity prior).
    use_analytical_grad : bool
        False (default) → true gradient via autograd + TBPTT (torch).
        True  → fast delta-rule heuristic gradient (numpy/numba, CPU only).
    tbptt_k : int or None
        Steps from the end through which gradients flow (autograd path).
        None → auto-computed as int(tbptt_frac * Nsim_opt).
    tbptt_frac : float
        Fraction of Nsim_opt for auto tbptt_k. Default 0.4.
    teq_opt : float or None
        Short equilibration (s) used during training iterations only.
        Full teq restored for final evaluation. Default 60 s.
    scheduler_patience : int
        ReduceLROnPlateau patience (consecutive non-improving iterations).
    n_iterations : int
        Number of optimisation iterations.
    n_simulations : int
        Independent forward passes averaged per gradient estimate (2-3 recommended).
    lr, min_lr : float
        Initial and minimum learning rate.
    grad_clip : float
        Gradient clipping threshold. Analytical: element-wise clamp.
        Autograd: L2-norm clip (preserves direction).
    converg_tol : float
        Convergence tolerance for early stopping (reserved, not yet active).
    weight_decay : float
        L2 regularisation (AdamW / SGD weight_decay). Default 1e-5.
    optimizer_type : str
        'adam' (default) or 'sgd'.
    momentum : float
        SGD momentum (ignored for Adam).
    max_gec : float or None
        Hard clamp on |GEC| weight after each step.
    gec_init_max : float or None
        Scales SC so max(GEC) = gec_init_max at initialisation.
    allow_negative : bool
        Allow negative GEC weights (inhibitory connections).
    verbose : bool
        Print loss / FC / gradient stats every 10 iterations.
    G, a, w, F0, F0_w : model parameters (see HopfModel)
    dt, teq, tmax, downsamp : timing for final evaluation run.
    dt_opt, teq_opt, tmax_opt, downsamp_opt : timing overrides for training.
    beta, sigma : noise parameters (see HopfModel).
    use_filt, low_cut, high_cut, filter_order : bandpass filter settings.
    full_output : bool
        Run a final full-tmax simulation and return extended diagnostics.
    device : str
        PyTorch compute device ('cpu' or 'cuda').

    Returns
    -------
    results : dict
        Always:
          'optimized_GEC'  : np.ndarray — fitted GEC (nnodes x nnodes)
          'fc_history'     : list[float] — FC correlation per iteration
          'loss_history'   : list[float] — MSE loss per iteration
        If full_output=True, also:
          'optimized_FC'      : np.ndarray — simulated FC at optimized GEC
          'empirical_FC'      : np.ndarray — input empirical FC
          'fc_similarity'     : float      — Pearson r(FC_emp, FC_sim) on triu
          'n_negative'        : int        — number of negative GEC weights
          'pct_negative'      : float      — % of active weights < 0
          'avg_weight_pos'    : float      — mean positive GEC weight
          'avg_weight_neg'    : float      — mean negative GEC weight
          'optimized_results' : np.ndarray — raw simulation (time x nodes x 2)
          'time_vector'       : np.ndarray — time axis for optimized_results
    """

    start_time = time.time()
    device = torch.device(device)
    nnodes = empirical_FC.shape[0]
    empirical_FC_np = np.asarray(empirical_FC, dtype=np.float64)
    triu_idx_np = np.triu_indices(nnodes, k=1)

    _dt = dt_opt if dt_opt is not None else dt
    _tmax = tmax_opt if tmax_opt is not None else tmax
    _downsamp = downsamp_opt if downsamp_opt is not None else downsamp
    _teq = float(teq_opt) if teq_opt is not None else float(teq)

    # Auto-compute TBPTT window
    _Nmax_opt = int(_tmax / _dt / _downsamp)
    _Neq_opt  = int(_teq  / _dt / _downsamp)
    _Nsim_opt = (_Nmax_opt + _Neq_opt - 1) * _downsamp + 1
    _tbptt_k  = tbptt_k if tbptt_k is not None else max(1, int(tbptt_frac * _Nsim_opt))

    C_np = C if C is not None else np.zeros((nnodes, nnodes))
    mask_np = (np.where(C_np > 0, 1.0, 0.0) if use_sc_mask
                else np.ones((nnodes, nnodes)))
    np.fill_diagonal(mask_np, 0)

    if verbose:
        if C is not None:
            print(f"Initial SC stats: max={C_np.max():.4f}, "
                  f"mean={C_np.mean():.4f}, "
                  f"sparsity={100.0 * (C_np == 0).sum() / (nnodes**2):.1f}%",
                  flush=True)
        else:
            print("No SC provided; initializing GEC with small random weights.",
                  flush=True)

    # Scale SC so initial GEC has a sensible weight range
    if gec_init_max is not None and C is not None:
        c_max = float(C_np.max())
        C_for_init = C_np * (gec_init_max / c_max) if c_max > 0 else C_np
    else:
        C_for_init = C_np

    empirical_FC_t = torch.tensor(empirical_FC_np, dtype=torch.float32, device=device)
    n_mask = max(float(
        np.where(C_np > 0, 1.0, 0.0).sum() if use_sc_mask
        else (nnodes * nnodes - nnodes)), 1.0)

    # Model
    model = HopfModel(
        nnodes=nnodes, C=C_for_init,
        backend='torch',
        G=G, a=a, w=w, F0=F0, F0_w=F0_w,
        beta=beta, sigma=sigma,
        dt=_dt, teq=_teq, tmax=_tmax, downsamp=_downsamp,
        use_filt=use_filt, low_cut=low_cut, high_cut=high_cut,
        filter_order=filter_order,
        tbptt_k=_tbptt_k,
        device=device.type
    )

    triu_indices = torch.triu_indices(nnodes, nnodes, offset=1, device=device)
    triu_row, triu_col = triu_indices[0], triu_indices[1]
    mask_t = torch.tensor(mask_np, dtype=torch.float32, device=device)

    if use_analytical_grad:
        # Numpy model for fast delta-rule forward passes
        model_np = HopfModel(
            nnodes=nnodes, C=C_for_init, backend='numpy',
            G=G, a=a, w=w, F0=F0, F0_w=F0_w,
            beta=beta, sigma=sigma,
            dt=_dt, teq=_teq, tmax=_tmax, downsamp=_downsamp,
            use_filt=use_filt, low_cut=low_cut, high_cut=high_cut,
            filter_order=filter_order
        )
    else:
        model_np = None

    if verbose:
        backend_str = ("numpy/numba" if use_analytical_grad
                       else f"torch ({device})")
        grad_str = "analytical" if use_analytical_grad else "autograd"
        print(f"Running optimisation ({grad_str} gradient, {backend_str} forward).",
              flush=True)
        if device.type == 'cuda' and not use_analytical_grad:
            print(f"GPU: {torch.cuda.get_device_name()}", flush=True)
        if tmax_opt is not None:
            print(f"Optimising with tmax_opt={tmax_opt}s, teq_opt={_teq}s "
                  f"(full tmax={tmax}s, teq={teq}s used for final eval).",
                  flush=True)
        if not use_analytical_grad:
            pct = 100.0 * _tbptt_k / _Nsim_opt
            print(f"TBPTT: tbptt_k={_tbptt_k} / Nsim_opt={_Nsim_opt} "
                  f"({pct:.0f}% coverage), n_sims={n_simulations} "
                  f"[stacked gradient]", flush=True)

    if optimizer_type.lower() == 'adam':
        optimizer = optim.AdamW([model.GEC], lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD([model.GEC], lr=lr, momentum=momentum,
                              weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=scheduler_patience,
        factor=0.5, min_lr=min_lr)

    FC_history   = []
    loss_history = []

    for iteration in range(n_iterations):

        optimizer.zero_grad()

        if use_analytical_grad:
            gec_np = model.GEC.detach().cpu().numpy().astype(np.float64) * mask_np
            if not allow_negative:
                gec_np = np.maximum(gec_np, 0.0)

            sim_corr_np = np.zeros((nnodes, nnodes))
            for _ in range(n_simulations):
                model_np.GEC = gec_np
                _res, _ = model_np.forward()
                x_sim = _res[:, :, 0]
                sim_corr_np += np.corrcoef(x_sim.T) / n_simulations

            sim_corr_np = np.clip(sim_corr_np, -1.0 + 1e-8, 1.0 - 1e-8)
            np.fill_diagonal(sim_corr_np, 1.0)

            diff_corr = sim_corr_np - empirical_FC_np
            loss_val = float(np.mean(diff_corr[triu_idx_np] ** 2))
            grad_np = np.clip(diff_corr * mask_np, -grad_clip, grad_clip)

            loss = torch.tensor(loss_val, dtype=torch.float32, device=device)
            simulated_FC = torch.tensor(sim_corr_np, dtype=torch.float32, device=device)
            model.GEC.grad = torch.tensor(grad_np, dtype=torch.float32, device=device)

        else:
            # Stacked gradient with single backward
            sim_corr_triu_list = []
            sim_metric_t = torch.zeros(nnodes, nnodes, device=device)

            for _ in range(n_simulations):
                _res, _ = model(mask_t, allow_negative)
                x_sim = _res[:, :, 0]
                sim_corr_i = torch.corrcoef(x_sim.T)
                sim_corr_triu_list.append(sim_corr_i[triu_row, triu_col])
                with torch.no_grad():
                    sim_metric_t += sim_corr_i.detach() / n_simulations

            fc_mean_triu = torch.stack(sim_corr_triu_list).mean(0)
            loss = torch.mean(
                (fc_mean_triu - empirical_FC_t[triu_row, triu_col]) ** 2)
            loss.backward()
            loss = loss.detach()

            # L2 norm clipping preserves gradient direction
            torch.nn.utils.clip_grad_norm_([model.GEC], max_norm=grad_clip)

            with torch.no_grad():
                simulated_FC = sim_metric_t

        # Gradient step and post-processing
        optimizer.step()
        with torch.no_grad():
            model.GEC.data *= mask_t # re-zero entries outside SC mask
            if not allow_negative:
                model.GEC.data.clamp_(min=0)
            if max_gec is not None:
                model.GEC.clamp_(-max_gec if allow_negative else 0, max_gec)
        scheduler.step(loss.item())

        # Tracking
        with torch.no_grad():
            fc_similarity = torch.corrcoef(
                torch.stack([empirical_FC_t[triu_row, triu_col],
                             simulated_FC[triu_row, triu_col]]))[0, 1]
        FC_history.append(fc_similarity.item())
        loss_history.append(loss.item())

        if verbose and iteration % 10 == 0:
            mins = (time.time() - start_time) / 60
            gec_abs = model.GEC.data.abs()
            gec_max = gec_abs.max().item()
            gec_mean = (gec_abs[mask_t > 0].mean().item()
                        if mask_t.any() else gec_abs.mean().item())
            print(f"Iteration {iteration}: Loss={loss.item():.4e}, "
                  f"FC corr={fc_similarity:.4f}, "
                  f"Grad max={model.GEC.grad.abs().max().item():.4e}, "
                  f"GEC |max/mean|={gec_max:.4f}/{gec_mean:.4f}, "
                  f"t={mins:.3f} min", flush=True)

        if (iteration > 100
                and torch.std(torch.tensor(loss_history[-10:])) < converg_tol):
            print(f"Converged at iteration {iteration}", flush=True)
            break

    optimized_GEC = model.GEC.detach().cpu().numpy() * mask_np

    if full_output:
        print("Running final simulation with optimized GEC...", flush=True)

        if use_analytical_grad:
            model_np.GEC = optimized_GEC
            model_np.tmax = tmax
            model_np.teq = teq
            model_np.dt = dt
            model_np.downsamp = downsamp
            opt_results, time_vector_np = model_np.forward()
            final_FC = np.corrcoef(opt_results[:, :, 0].T)
            emp_triu = empirical_FC_np[triu_idx_np]
            sim_triu = final_FC[triu_idx_np]
            fc_sim_val = float(np.corrcoef(emp_triu, sim_triu)[0, 1])
            opt_out = opt_results
            tvec_out = time_vector_np
        else:
            if any(x is not None for x in [tmax_opt, dt_opt, downsamp_opt, teq_opt]):
                model.tmax = tmax
                model.teq = teq
                model.dt = dt
                model.downsamp = downsamp
                model.dt_tensor = torch.tensor(dt, dtype=torch.float32, device=device)
            with torch.no_grad():
                opt_results, time_vector_t = model(mask_t, allow_negative)
            final_FC = torch.corrcoef(opt_results[:, :, 0].T).detach().cpu().numpy()
            emp_triu = empirical_FC_np[triu_idx_np]
            sim_triu = final_FC[triu_idx_np]
            fc_sim_val = float(np.corrcoef(emp_triu, sim_triu)[0, 1])
            opt_out = opt_results.detach().cpu().numpy()
            tvec_out = time_vector_t.detach().cpu().numpy()

        n_connections = int(np.sum(optimized_GEC != 0))
        n_negative = int(np.sum(optimized_GEC < 0))
        pct_negative = (n_negative / n_connections * 100) if n_connections > 0 else 0.0
        GEC_pos = np.maximum(optimized_GEC, 0)
        GEC_neg = np.maximum(-optimized_GEC, 0)
        avg_weight_pos = float(np.mean(GEC_pos[GEC_pos > 0])) if np.any(GEC_pos > 0) else 0.0
        avg_weight_neg = float(np.mean(GEC_neg[GEC_neg > 0])) if np.any(GEC_neg > 0) else 0.0

        return {
            "optimized_GEC":     optimized_GEC,
            "optimized_FC":      final_FC,
            "empirical_FC":      empirical_FC_np,
            "fc_similarity":     fc_sim_val,
            "fc_history":        FC_history,
            "loss_history":      loss_history,
            "n_negative":        n_negative,
            "pct_negative":      pct_negative,
            "avg_weight_pos":    avg_weight_pos,
            "avg_weight_neg":    avg_weight_neg,
            "optimized_results": opt_out,
            "time_vector":       tvec_out,
        }

    return {
        "optimized_GEC": optimized_GEC,
        "fc_history":    FC_history,
        "loss_history":  loss_history,
    }



def qplot(BOLD, dt=1, downsamp=1, 
          vmin_fc=-1, vmax_fc=1,
          fmin=0, fmax=4):
    """
    Quick summary plot: time traces, power spectrum, and FC matrix.

    Parameters
    ----------
    BOLD : np.ndarray  (nnodes x time)
        BOLD-like signal (x-component of oscillator state).
    dt : float
        Integration timestep (s).
    downsamp : int
        Downsampling factor used during simulation.
    fmin : float
        Min freq for PSD plot (Hz).
    fmax:
        Max frew for PSD plot (Hz).

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    nnodes = BOLD.shape[0]
    fig, ax = plt.subplots(1, 3, figsize=(15, 3),
                           gridspec_kw={'width_ratios': [3, 1, 1]})

    # Time traces (up to 5 random nodes)
    for i in np.random.choice(nnodes, min(5, nnodes), replace=False):
        ax[0].plot(BOLD[i, :], color='black', linewidth=0.5)
    ax[0].set_xlabel('Time (steps)')
    ax[0].set_ylabel('Amplitude (a.u.)')

    # Power spectral density
    fs = 1 / (dt * downsamp)
    f, Pxx = signal.welch(BOLD.T, fs=fs, nperseg=1024, axis=0)
    Pxx_mean = np.mean(Pxx, axis=1)
    Pxx_sem  = np.std(Pxx,  axis=1) / np.sqrt(nnodes)
    ax[1].semilogy(f, Pxx_mean, color='black')
    ax[1].fill_between(f, Pxx_mean - Pxx_sem, Pxx_mean + Pxx_sem,
                       color='gray', alpha=0.5)
    ax[1].set_xlim(fmin, fmax)
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('PSD')

    # FC matrix
    corr_matrix = np.corrcoef(BOLD)
    im = ax[2].imshow(corr_matrix, vmin=vmin_fc, vmax=vmax_fc,
                      cmap='bwr', aspect='equal')
    fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04, label='FC')
    ax[2].set_xlabel('Node')
    ax[2].set_ylabel('Node')

    plt.tight_layout()
    return fig, ax




# Example simulation
if __name__ == "__main__":

    ex_param = _example_params()
    model = HopfModel(backend='numpy', **ex_param)
    sim_results, _ = model.forward(verbose=True)
    ex_BOLD = model.to_BOLD(sim_results)
    fig, _ = qplot(ex_BOLD)
    plt.show()
    
