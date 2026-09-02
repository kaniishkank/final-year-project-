"""
EviGuard AI Proctoring System Launcher
Starts the interactive Streamlit dashboard.
"""

import os
import subprocess
import sys

def main():
    print("=" * 60)
    print("  🛡️  Starting EviGuard AI Proctoring & Evidence Analysis System")
    print("=" * 60)
    
    dashboard_path = os.path.join(os.path.dirname(__file__), "frontend", "dashboard.py")
    
    cmd = [sys.executable, "-m", "streamlit", "run", dashboard_path]
    print(f"Executing: {' '.join(cmd)}\n")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[EviGuard] Server stopped by user.")

if __name__ == "__main__":
    main()
