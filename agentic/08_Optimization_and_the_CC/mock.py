"""HOD + IA mock on BolPlanck z=0, and its data vector d = [w_gg, w_g+].

Conventions (see IA_Without_Gradients_Plan.md):
  Zheng07 threshold Mr < -21, NFW satellites, CentralAlignment(mu_c),
  RadialSatelliteAlignment(mu_s). Line of sight = z, real space, e = 1 for every
  galaxy (the w_g+ normalization is absorbed into C1 later). Units Mpc/h.
"""
import numpy as np
from halotools.empirical_models import (HodModelFactory, TrivialPhaseSpace, NFWPhaseSpace,
                                        Zheng07Cens, Zheng07Sats)
from halotools.empirical_models.ia_models.ia_model_components import (
    CentralAlignment, RadialSatelliteAlignment)
from halotools.mock_observables import wp
from halotools.mock_observables.ia_correlations import gi_plus_projected
from halotools.sim_manager import CachedHaloCatalog
from halotools.empirical_models.ia_models import (ia_model_components, ia_strength_models,
                                                  watson_distribution)
from halotools.utils import mcrotations
from astropy.utils.misc import NumpyRNGContext


class _SeedFromGlobal(NumpyRNGContext):
    """NumpyRNGContext(None) reseeds from OS entropy, so the alignment draws can't be
    reproduced. Draw the seed from the global state instead, which populate() seeds."""
    def __init__(self, seed=None):
        super().__init__(np.random.randint(2**31) if seed is None else seed)


for _module in (ia_model_components, ia_strength_models, watson_distribution, mcrotations):
    _module.NumpyRNGContext = _SeedFromGlobal

THRESHOLD = -21
RP_BINS = np.logspace(np.log10(0.5), np.log10(25.0), 16)   # 15 bins
RP = np.sqrt(RP_BINS[1:] * RP_BINS[:-1])
PI_MAX = 30.0
NBIN = len(RP)

# Fitted parameters, in order.  The HOD values are Zheng07's Mr < -21 defaults.
NAMES = ['logMmin', 'sigma_logM', 'alpha', 'logM1',
         'central_alignment_strength', 'satellite_alignment_strength']
THETA_TRUE = np.array([12.79, 0.39, 1.15, 13.94, 0.8, 0.5])


def load_halos():
    """BolPlanck z=0 Rockstar, dropping halos whose shape axis is undefined (all zeros)."""
    halocat = CachedHaloCatalog(simname='bolplanck', halo_finder='rockstar',
                                redshift=0, version_name='halotools_v0p4')
    t = halocat.halo_table
    bad = (t['halo_axisA_x'] == 0) & (t['halo_axisA_y'] == 0) & (t['halo_axisA_z'] == 0)
    halocat._halo_table = t[~bad]
    return halocat


def build_model(halocat, theta=THETA_TRUE, seed=0):
    model = HodModelFactory(
        centrals_occupation=Zheng07Cens(threshold=THRESHOLD),
        centrals_profile=TrivialPhaseSpace(),
        satellites_occupation=Zheng07Sats(threshold=THRESHOLD, modulate_with_cenocc=True),
        satellites_profile=NFWPhaseSpace(),
        centrals_orientation=CentralAlignment(),
        satellites_orientation=RadialSatelliteAlignment(Lbox=halocat.Lbox),
        model_feature_calling_sequence=('centrals_occupation', 'centrals_profile',
                                        'satellites_occupation', 'satellites_profile',
                                        'centrals_orientation', 'satellites_orientation'))
    set_theta(model, theta)
    model.populate_mock(halocat, seed=seed)
    return model


def set_theta(model, theta):
    for name, value in zip(NAMES, theta):
        assert name in model.param_dict, name
        model.param_dict[name] = value


def populate(model, theta, seed):
    set_theta(model, theta)
    np.random.seed(seed)   # the alignment draws ignore `seed`; see _SeedFromGlobal
    model.mock.populate(seed=seed)
    return model.mock.galaxy_table


def measure(gals, Lbox, num_threads=1):
    """Data vector [w_gg (15), w_g+ (15)] for one galaxy table."""
    pos = np.vstack([gals['x'], gals['y'], gals['z']]).T % Lbox
    ori = np.vstack([gals['galaxy_axisA_x'], gals['galaxy_axisA_y']]).T
    e = np.ones(len(gals))
    w_gg = wp(pos, RP_BINS, PI_MAX, period=Lbox, num_threads=num_threads)
    w_gp = gi_plus_projected(pos, ori, e, pos, RP_BINS, PI_MAX, period=Lbox,
                             num_threads=num_threads)
    return np.concatenate([w_gg, w_gp])
