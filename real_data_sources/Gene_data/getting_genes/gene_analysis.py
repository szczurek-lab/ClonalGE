import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy import sparse


sparse_Y = sparse.load_npz('files_variables/breast_Y_raw.npz')
Y = sparse_Y.toarray()