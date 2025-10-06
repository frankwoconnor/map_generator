import os
import sys
import subprocess
from datetime import datetime

MAIN_SCRIPT = 'main.py'

def main():
    """Regenerate a map from a specified config.json file."""
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <path_to_config.json>", file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1]
    if not os.path.isfile(config_path):
        print(f"Error: Configuration file not found at '{config_path}'", file=sys.stderr)
        sys.exit(1)

    # Generate a new timestamped prefix for this regeneration run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"regen_{timestamp}"

    print(f"--- Starting Map Regeneration ---")
    print(f"Config File: {config_path}")
    print(f"Output Prefix: {prefix}")
    print(f"---------------------------------")

    # Construct the command to run main.py
    cmd = [
        'python3',
        MAIN_SCRIPT,
        '--config',
        config_path,
        '--prefix',
        prefix
    ]

    print(f"Executing command: {' '.join(cmd)}")

    # Execute the main script as a subprocess
    try:
        # We stream the output directly to the console
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                print(line, end='')
        
        process.wait()
        return_code = process.returncode

        if return_code == 0:
            print("\n--- Map Regeneration Successful ---")
        else:
            print(f"\n--- Map Regeneration Failed (Exit Code: {return_code}) ---", file=sys.stderr)
        
        sys.exit(return_code)

    except FileNotFoundError:
        print(f"Error: '{MAIN_SCRIPT}' not found. Make sure you are in the correct directory.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
