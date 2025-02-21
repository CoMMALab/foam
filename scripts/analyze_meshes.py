import os
import csv
import time
import signal
import trimesh
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
from threading import Thread
import sys
import queue
import tqdm

def ensure_output_directory(script_dir: Path) -> Path:
    output_dir = script_dir / 'outputs'
    output_dir.mkdir(exist_ok=True)
    return output_dir

def get_mesh_files(directory: str) -> List[Path]:
    valid_extensions = {'.stl', '.dae', '.obj'}
    directory_path = Path(directory)
    mesh_files = []

    for root, _, files in os.walk(directory_path):
        root_path = Path(root)
        for file in files:
            file_path = root_path / file
            if file_path.suffix.lower() in valid_extensions:
                mesh_files.append(file_path)

    return mesh_files

def get_triangle_count(mesh_path: Path) -> int:
    try:
        mesh = trimesh.load(str(mesh_path))
        return len(mesh.faces)
    except Exception as e:
        print(f"Error loading mesh {mesh_path}: {e}")
        return -1

def input_thread(user_input_queue):
    while True:
        user_input = input()
        if user_input.strip() == "":
            user_input_queue.put("skip")
            break

def process_mesh(mesh_path: Path, output_dir: Path) -> Optional[Tuple[str, float, int, int]]:
    print(f"\nProcessing {mesh_path}...")
    print("Press Enter at any time to skip this mesh...")
    
    # Get triangle count
    triangle_count = get_triangle_count(mesh_path)
    if triangle_count == -1:
        return None
    
    # Create output filename
    output_json = output_dir / f"{mesh_path.stem}-spheres.json"

    # Create a queue for user input
    user_input_queue = queue.Queue()

    # Start input monitoring thread
    input_monitor = Thread(target=input_thread, args=(user_input_queue,), daemon=True)
    input_monitor.start()

    # Run generate_spheres.py
    start_time = time.time()
    try:
        process = subprocess.Popen(
            ['python3', 'generate_spheres.py', str(mesh_path), '--output', str(output_json)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        while process.poll() is None:
            # Check if user wants to skip
            try:
                if not user_input_queue.empty() and user_input_queue.get_nowait() == "skip":
                    print("\nSkipping this mesh...")
                    process.kill()
                    return None
            except queue.Empty:
                pass

            time.sleep(0.1)

        if process.returncode == 0:
            processing_time = time.time() - start_time

            # Get output
            stdout, _ = process.communicate()

            # Parse the output to get sphere count
            sphere_count = -1
            for line in stdout.split('\n'):
                if "Total number of spheres:" in line:
                    try:
                        sphere_count = int(line.split(":")[-1].strip())
                    except ValueError:
                        continue

            if sphere_count != -1:
                return (str(mesh_path), processing_time, sphere_count, triangle_count)

    except Exception as e:
        print(f"Error processing {mesh_path}: {e}")
        return None

    return None

def main():
    # Get the script directory and create outputs directory
    script_dir = Path(__file__).parent
    output_dir = ensure_output_directory(script_dir)
    
    # Get all mesh files recursively
    mesh_files = get_mesh_files(script_dir)
    
    if not mesh_files:
        print("No mesh files found in the current directory or its subdirectories!")
        return
    
    print(f"Found {len(mesh_files)} mesh files")
    print(f"JSON outputs will be saved to: {output_dir}")

    # Process each mesh and collect results
    results = []
    processed_count = 0
    skipped_count = 0
    error_count = 0

    for mesh_file in tqdm.tqdm(mesh_files):
        result = process_mesh(mesh_file, output_dir)
        if result is not None:
            results.append(result)
            processed_count += 1
        else:
            skipped_count += 1
    
    if results:
        # Write results to CSV in the output directory
        csv_path = output_dir / 'mesh_analysis_results.csv'
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['File Path', 'Processing Time (s)', 'Sphere Count', 'Triangle Count'])
            writer.writerows(results)
        print(f"\nResults written to {csv_path}")
    
    print(f"\nAnalysis complete!")
    print(f"Successfully processed: {processed_count} meshes")
    print(f"Skipped/Failed: {skipped_count} meshes")

if __name__ == "__main__":
    main()