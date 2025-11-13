# credit_card_fraud_detection/run_dashboard.py
"""
Quick start script for the Streamlit dashboard
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['streamlit', 'plotly', 'matplotlib', 'seaborn']
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_missing_packages(missing_packages):
    """Install missing packages"""
    for package in missing_packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    """Main function to run the dashboard"""
    print("🚀 Starting Fraud Detection Dashboard...")
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print(f"Missing packages: {missing}")
        install_missing_packages(missing)
    
    # Run the Streamlit app
    app_path = Path(__file__).parent / "app.py"
    
    print("📊 Launching Streamlit dashboard...")
    print("👉 The dashboard will open in your browser automatically")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    
    try:
        subprocess.run(["streamlit", "run", str(app_path)], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")

if __name__ == "__main__":
    main()