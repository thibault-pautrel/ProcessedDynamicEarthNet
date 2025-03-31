import subprocess
import time
import datetime
import os

def log_message(log_file, message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"[{timestamp}] {message}\n")

def run_command(command, log_file):
    log_message(log_file, f"Starting command: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(log_file, 'a') as f:
        for line in process.stdout:
            print(line, end='')
            f.write(line)
    process.wait()
    log_message(log_file, f"Command finished with return code: {process.returncode}\n")

def main():
    logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Define the runs: first unet_pipeline.py, then basic_spdnet_pipeline.py
    runs = [
        {
            'name': 'UNet',
            'command': 'python3 scripts/unet_pipeline.py',
            'log': os.path.join(logs_dir, 'unet.log')
        },
        {
            'name': 'SPDNet',
            'command': 'python3 scripts/basic_spdnet_pipeline.py',
            'log': os.path.join(logs_dir, 'spdnet.log')
        }
    ]
    
    for run in runs:
        print(f"\n=== Starting {run['name']} ===")
        log_message(run['log'], f"=== Starting {run['name']} ===")
        run_command(run['command'], run['log'])
        log_message(run['log'], f"=== Finished {run['name']} ===\n\n")
        print(f"=== Finished {run['name']} ===\n")
        time.sleep(10)  # Wait 10 seconds between runs

    print("All weekend runs completed.")

if __name__ == "__main__":
    main()
