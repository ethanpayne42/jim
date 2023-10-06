from jaxtyping import Array
from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc
from ripple.waveforms.IMRPhenomPv2 import gen_IMRPhenomPv2_hphc
from jaxNRSur.SurrogateModel import SurrogateModel
import jax.numpy as jnp
from abc import ABC
from scipy.signal.windows import tukey

# geometric units to SI
GMSUN_SI = 1.32712442099000e+20
C_SI = 2.99792458000000e+08
RSUN_SI = GMSUN_SI / C_SI**2

# parsecs to SI
PC_SI = 3.08567758149136720000e+16
MPC_SI = 1E6*PC_SI


class Waveform(ABC):
    def __init__(self):
        return NotImplemented

    def __call__(self, axis: Array, params: Array) -> Array:
        return NotImplemented


class RippleIMRPhenomD(Waveform):

    f_ref: float

    def __init__(self, f_ref: float = 20.0):
        self.f_ref = f_ref

    def __call__(self, frequency: Array, params: dict) -> dict:
        output = {}
        ra = params["ra"]
        dec = params["dec"]
        theta = [
            params["M_c"],
            params["eta"],
            params["s1_z"],
            params["s2_z"],
            params["d_L"],
            0,
            params["phase_c"],
            params["iota"],
        ]
        hp, hc = gen_IMRPhenomD_hphc(frequency, theta, self.f_ref)
        output["p"] = hp
        output["c"] = hc
        return output


class RippleIMRPhenomPv2(Waveform):

    f_ref: float

    def __init__(self, f_ref: float = 20.0):
        self.f_ref = f_ref

    def __call__(self, frequency: Array, params: dict) -> Array:
        output = {}
        theta = [
            params["M_c"],
            params["eta"],
            params['s1_x'],
            params['s1_y'],
            params["s1_z"],
            params['s2_x'],
            params['s2_y'],
            params["s2_z"],
            params["d_L"],
            0,
            params["phase_c"],
            params["iota"],
        ]
        hp, hc = gen_IMRPhenomPv2_hphc(frequency, theta, self.f_ref)
        output["p"] = hp
        output["c"] = hc
        return output


class NRHybSur3dq8FD(Waveform):
    
    def __init__(self, datapath: str, seglen: float, srate: float, tukey_alpha: float=0.4):
        self.model = SurrogateModel(datapath)
        self.tukey_alpha = tukey_alpha
        
        # Precomputing the time and window arrays
        N = int(seglen*srate)
        self.time = jnp.arange(N)/srate - seglen + 2
        self.window = tukey(N, self.tukey_alpha)
        self.n = N//2 + 1
        
    def __call__(self, frequency: Array, params: dict) -> Array:
        
        output = {}
        
        # Set up the parameters
        eta = params["q"]/(1+params["q"])**2
        M = params["M_c"]/eta**0.6
        theta_NRHyb = jnp.array([params["q"], params["s1_z"], params["s2_z"]])
        
        # Convert to geometric unit time array for NRSur
        time_m = self.time * C_SI / RSUN_SI / M
        
        hrM_TD = self.model.get_waveform(
            time_m, theta_NRHyb, theta=params["iota"], phi=params["phase_c"]
        )
        
        # Fourier transform
        hrM_FD = jnp.fft.fft(hrM_TD * self.window) /frequency[-1]/2
        
        # get FD plus and cross polarizations (following Max's calculation)
        h_fd_positive = hrM_FD[:self.n]
        conj_h_fd_negative = jnp.conj(jnp.fft.ifftshift(hrM_FD))[:self.n][::-1]
        
        hp_rM_fd = (h_fd_positive + conj_h_fd_negative)/2
        hc_rM_fd = 1j*(h_fd_positive - conj_h_fd_negative)/2
        
        # hp_rM_fd = jnp.fft.rfft(hrM_TD.real * self.window) /frequency[-1]/2
        # hc_rM_fd = jnp.fft.rfft(-hrM_TD.imag * self.window) /frequency[-1]/2
        
        # this is h * r / M, so scale by the mass and distance        
        hp_fd = hp_rM_fd * M * RSUN_SI / params["d_L"] / MPC_SI
        hc_fd = hc_rM_fd * M * RSUN_SI / params["d_L"] / MPC_SI
        
        output["p"] = hp_fd
        output["c"] = hc_fd
        return output