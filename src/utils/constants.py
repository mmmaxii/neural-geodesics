import numpy as np

# Unidades Naturales
G = 1.0
C = 1.0

# Masa por defecto
DEFAULT_M = 1.0

# Factores de radio respecto a r_s (r_s = 2M)
RS_FACTOR = 2.0                     # r_s = 2 * M
PHOTON_SPHERE_FACTOR = 1.5          # r_photon = 1.5 * r_s = 3M
ISCO_FACTOR = 3.0                   # r_isco = 3 * r_s = 6M
B_CRIT_FACTOR = (3.0 * np.sqrt(3.0)) / 2.0  # b_crit ≈ 2.598076 * r_s

# Configuración por defecto de Rendering / Grid
DEFAULT_RESOLUTION = 512
DEFAULT_FOV_RS = 15.0