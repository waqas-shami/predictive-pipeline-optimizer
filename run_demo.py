#!/usr/bin/env python
"""
Run the Predictive Pipeline Optimizer Streamlit Demo

Usage:
    python run_demo.py

Or directly:
    streamlit run app/streamlit_app.py
"""

import subprocess
import sys


def main():
    """Launch the Streamlit demo application."""
    print("=" * 60)
    print("  Predictive Pipeline Optimizer")
    print("  AI-Powered Failure Prediction & Optimization")
    print("=" * 60)
    print()
    print("Starting Streamlit server...")
    print("The browser will open automatically.")
    print()
    print("Press Ctrl+C to stop the server.")
    print("=" * 60)

    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"],
            check=True
        )
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except subprocess.CalledProcessError as e:
        print(f"\nError starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
