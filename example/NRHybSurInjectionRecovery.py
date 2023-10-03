from jimgw.jim import Jim
from jimgw.detector import H1, L1, V1
from jimgw.likelihood import TransientLikelihoodFD
from jimgw.waveform import NRHybSur3dq8FD
from jimgw.prior import Uniform
from ripple import ms_to_Mc_eta
import jax.numpy as jnp
import jax
from astropy.time import Time

from tap import Tap
import yaml
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)


class InjectionRecoveryParser(Tap):
    config: str 
    
    # Noise parameters
    seed: int = 0
    f_sampling: int  = 4096
    duration: int = 4
    fmin: float = 20.0
    ifos: list[str]  = ["H1", "L1", "V1"]

    # Injection parameters
    m1: float = 80.0
    m2: float = 70.0
    s1_z: float = 0.5
    s2_z: float = 0.5
    dist_mpc: float = 4000.
    tc: float = 0.
    #phic: float = 0.0
    inclination: float = 0.5
    polarization_angle: float = 0.7
    ra: float = 1.2
    dec: float = 0.3

    # Sampler parameters
    n_dim: int = 10
    n_chains: int = 500
    n_loop_training: int = 200
    n_loop_production: int = 10
    n_local_steps: int = 300
    n_global_steps: int = 300
    learning_rate: float = 0.001
    max_samples: int = 60000
    momentum: float = 0.9
    num_epochs: int = 300
    batch_size: int = 60000
    stepsize: float = 0.01
    use_global: bool = True
    keep_quantile: float = 0.0
    train_thinning: int = 1
    output_thinning: int = 30
    num_layers: int = 6
    hidden_size: list[int] = [64,64]
    num_bins: int = 8

    # Output parameters
    output_path: str = "./"
    downsample_factor: int = 10


args = InjectionRecoveryParser().parse_args()

# Fetch noise parameters 

print("Constructing detectors")
print("Making noises")

#Fetch injection parameters and inject signal

print("Injection signals")

freqs = jnp.linspace(args.fmin, args.f_sampling/2, args.duration*(args.f_sampling//2)+1)

Mc, eta = ms_to_Mc_eta(jnp.array([args.m1, args.m2]))
trigger_time = 1126259462.4
post_trigger_duration = 2
epoch = args.duration - post_trigger_duration
gmst = Time(trigger_time, format='gps').sidereal_time('apparent', 'greenwich').rad

waveform = NRHybSur3dq8FD('/mnt/home/epayne/NRHybSur3dq8.h5', 
                          seglen=args.duration, srate=args.f_sampling)
prior = Uniform(
    xmin = [10, 0.125, 0, 0, 0., -0.05, -1, 0., 0.,-1.],
    xmax = [130., 1., 1., 1., 1e4, 0.05, 1., jnp.pi, 2*jnp.pi, 1.],
    naming = ["M_c", "q", "s1_z", "s2_z", "d_L", "t_c", "cos_iota", "psi", "ra", "sin_dec"],
    transforms = {"q": ("eta", lambda params: params['q']/(1+params['q'])**2),
                 "cos_iota": ("iota",lambda params: jnp.arccos(params['cos_iota'])),
                 "sin_dec": ("dec",lambda params: jnp.arcsin(params['sin_dec']))} # sin and arcsin are periodize cos_iota and sin_dec
)
true_param = jnp.array([Mc, args.m2/args.m1, args.s1_z, args.s2_z, args.dist_mpc, args.tc, jnp.cos(args.inclination), args.polarization_angle, args.ra, jnp.sin(args.dec)])
print(true_param)
true_param = prior.add_name(true_param, transform_name = True, transform_value = True)
print(true_param)
detector_param = {"ra": args.ra, "dec": args.dec, "gmst": gmst, "psi": args.polarization_angle, "epoch": epoch, "t_c": args.tc}
h_sky = waveform(freqs, true_param)
key, subkey = jax.random.split(jax.random.PRNGKey(args.seed+1234))
H1.inject_signal(subkey, freqs, h_sky, detector_param)
key, subkey = jax.random.split(key)
L1.inject_signal(subkey, freqs, h_sky, detector_param)
#key, subkey = jax.random.split(key)
#V1.inject_signal(subkey, freqs, h_sky, detector_param)

likelihood = TransientLikelihoodFD([H1, L1], waveform, trigger_time, args.duration, post_trigger_duration)

mass_matrix = jnp.eye(args.n_dim)
mass_matrix = mass_matrix.at[1,1].set(1e-3)
mass_matrix = mass_matrix.at[9,9].set(1e-3)
local_sampler_arg = {"step_size": mass_matrix*3e-3}

print((jnp.array(list(true_param.values()))))
print(likelihood.evaluate(true_param, {}))
print(waveform(freqs, true_param))
print(prior.log_prob(jnp.array(list(true_param.values())[:-1])))

jim = Jim(likelihood, 
        prior,
        n_loop_training=args.n_loop_training,
        n_loop_production = args.n_loop_production,
        n_local_steps=args.n_local_steps,
        n_global_steps=args.n_global_steps,
        n_chains=args.n_chains,
        n_epochs=args.num_epochs,
        learning_rate = args.learning_rate,
        max_samples = args.max_samples,
        momentum = args.momentum,
        batch_size = args.batch_size,
        use_global=args.use_global,
        keep_quantile= args.keep_quantile,
        train_thinning = args.train_thinning,
        output_thinning = args.output_thinning,
        local_sampler_arg = local_sampler_arg,
        seed = args.seed,
        num_layers = args.num_layers,
        hidden_size = args.hidden_size,
        num_bins = args.num_bins
        )

key, subkey = jax.random.split(key)
jim.sample(subkey)
samples = jim.get_samples()