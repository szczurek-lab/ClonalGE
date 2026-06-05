"""
clonalge — core library package.

Sub-modules
-----------
model                : ClonalGE Gibbs-sampler class (was clonalGE.py)
tumoroscope          : Tumoroscope baseline model
simulation           : Synthetic-data generator for ClonalGE
simulation_tumoroscope : Synthetic-data generator for Tumoroscope
visualization        : Plotting utilities
pre_processing       : Data-integration helpers (ST + WES → model inputs)
constants            : Global hyper-parameters and run settings
run_selection        : Cross-run / cross-chain consensus selection
select_chains        : Per-run chain selection helper
"""

from . import model
from . import tumoroscope
from . import simulation
from . import simulation_tumoroscope
from . import visualization
from . import pre_processing
from . import constants
from . import run_selection
from . import select_chains
