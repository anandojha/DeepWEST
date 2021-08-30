"""
DeepWEST

Deep learning approaches for Weighted Ensemble Simulations Toolkit  for faster and enhanced sampling of kinetics and thermodynamics
"""

# Add imports here
from .DeepWEST import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
