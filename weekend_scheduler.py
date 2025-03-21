import subprocess
import time
import datetime
import os
import glob

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

def get_common_planet_folders():
    spdnet_dir = "/media/thibault/DynEarthNet/datasets/spdnet"
    unet_dir = "/media/thibault/DynEarthNet/datasets/unet"
    
    spdnet_planets = {os.path.basename(p) for p in glob.glob(os.path.join(spdnet_dir, "planet*")) if os.path.isdir(p)}
    unet_planets = {os.path.basename(p) for p in glob.glob(os.path.join(unet_dir, "planet*")) if os.path.isdir(p)}
    
    common_planets = sorted(list(spdnet_planets.intersection(unet_planets)))
    return common_planets

def main():
    logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    planet_folders = get_common_planet_folders()
    if not planet_folders:
        print("No common planet folders found in both spdnet and unet directories.")
        return

    runs = []
    for planet_folder in planet_folders:
        # SPDNet run
        runs.append({
            'name': f'SPDNet_{planet_folder}',
            'command': f'python scripts/spdnet_pipeline.py --use_batch_norm False --planet_folder {planet_folder}',
            'log': os.path.join(logs_dir, f'spdnet_{planet_folder}.log')
        })
        # UNet run
        runs.append({
            'name': f'UNet_{planet_folder}',
            'command': f'python scripts/unet_pipeline.py --planet_folder {planet_folder}',
            'log': os.path.join(logs_dir, f'unet_{planet_folder}.log')
        })
    
    for run in runs:
        print(f"\n=== Starting {run['name']} ===")
        log_message(run['log'], f"=== Starting {run['name']} ===")
        run_command(run['command'], run['log'])
        log_message(run['log'], f"=== Finished {run['name']} ===\n\n")
        print(f"=== Finished {run['name']} ===\n")
        time.sleep(10)

    print("All weekend runs completed.")

if __name__ == "__main__":
    main()
