"""
short description of mdnet.

Machine Learning Analysis of Molecular Dynamics Trajectories for Weighted Ensemble simulations
"""

# Add imports here
from .mdnet import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
