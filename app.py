"""
EviGuard AI Proctoring System - Application Entrypoint
Launches the full interactive proctoring suite powered by EviGuardRobustEngine.
"""

import os
import sys

# Ensure root directory is in sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import runpy

dashboard_script = os.path.join(root_dir, "frontend", "dashboard.py")
runpy.run_path(dashboard_script, run_name="__main__")
