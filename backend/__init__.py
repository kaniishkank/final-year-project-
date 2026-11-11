"""
EviGuard AI Proctoring Backend Package
"""

import sys
import os

# Ensure package root is present in sys.path
_pkg_root = os.path.abspath(os.path.dirname(__file__))
_project_root = os.path.abspath(os.path.join(_pkg_root, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Explicitly bind subpackages to avoid PEP 420 namespace package KeyError in importlib
from . import db
from . import detection
from . import explainability
from . import pose
from . import reporting
from . import scoring
from . import tracking

__version__ = "2.5.0"
__all__ = [
    "db",
    "detection",
    "explainability",
    "pose",
    "reporting",
    "scoring",
    "tracking",
]
