"""Hardware adapters — bridge platform-specific data into Recording objects."""

from nuro.adapters.aer import from_aedat, from_aer_binary, from_aer_events
from nuro.adapters.file import from_csv, from_hdf5, from_numpy, from_nir_file
