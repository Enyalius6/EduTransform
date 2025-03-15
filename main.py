import subprocess
import sys

def run_script(script_name):
    print(f"\nRunning {script_name}...")
    # Use the current Python interpreter (sys.executable) to run the script.
    subprocess.run([sys.executable, script_name], check=True)
    print(f"{script_name} completed.")

if __name__ == "__main__":
    # List your scripts in the order you want them to run.
    run_script("datatovectors.py")
    run_script("mcqgenerator.py")
    run_script("quiz.py")
    run_script("content_generator.py")
