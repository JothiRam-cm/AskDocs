"""Run script for the Streamlit application."""
import subprocess
import sys
from pathlib import Path

def main():
    """Run the Streamlit application."""
    # Use relative path from the script location
    current_dir = Path(__file__).parent
    app_path = current_dir / "src" / "app" / "main.py"
    if not app_path.exists():
        print(f"Error: Could not find {app_path}")
        sys.exit(1)
        
    try:
        subprocess.run(["streamlit", "run", str(app_path)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 