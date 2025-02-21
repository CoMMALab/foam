import os
import time
import signal
import trimesh
import subprocess
import shutil
from pathlib import Path
from typing import List
from threading import Thread
import queue

def ensure_failed_meshes_directory(script_dir: Path) -> Path:
    """Create failed_meshes directory if it doesn't exist."""
    failed_dir = script_dir.parent / 'failed_meshes'
    failed_dir.mkdir(exist_ok=True)
    return failed_dir

def get_mesh_files(directory: str) -> List[Path]:
    """Recursively get all mesh files (stl, dae, obj) in the given directory and its subdirectories."""
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

def input_thread(user_input_queue):
    """Thread function to monitor for user input"""
    while True:
        user_input = input()
        if user_input.strip() == "":
            user_input_queue.put("skip")
            break

def test_mesh(mesh_path: Path, failed_dir: Path) -> bool:
    """
    Test if a mesh can be processed successfully.
    Returns True if mesh should be moved to failed directory.
    """
    print(f"\nTesting {mesh_path}...")
    print("Press Enter at any time to skip this mesh...")
    
    # Try loading the mesh first
    try:
        mesh = trimesh.load(str(mesh_path))
    except Exception as e:
        print(f"Error loading mesh {mesh_path}: {e}")
        return True
    
    # Create a queue for user input
    user_input_queue = queue.Queue()
    
    # Start input monitoring thread
    input_monitor = Thread(target=input_thread, args=(user_input_queue,), daemon=True)
    input_monitor.start()
    
    # Run generate_spheres.py just to test
    try:
        process = subprocess.Popen(
            ['python3', 'generate_spheres.py', str(mesh_path)],
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
                    return False
            except queue.Empty:
                pass
            
            time.sleep(0.1)
        
        # If process failed, return True to move the mesh
        if process.returncode != 0:
            _, stderr = process.communicate()
            print(f"Error processing mesh: {stderr}")
            return True
            
    except Exception as e:
        print(f"Error during processing: {e}")
        return True
    
    return False

def main():
    # Get the script directory and create failed_meshes directory
    script_dir = Path(__file__).parent
    failed_dir = ensure_failed_meshes_directory(script_dir)
    
    # Get all mesh files recursively
    mesh_files = get_mesh_files(script_dir)
    
    if not mesh_files:
        print("No mesh files found in the current directory or its subdirectories!")
        return
    
    print(f"Found {len(mesh_files)} mesh files")
    print(f"Failed meshes will be moved to: {failed_dir}")
    
    failed_count = 0
    skipped_count = 0
    success_count = 0
    
    for mesh_file in mesh_files:
        should_move = test_mesh(mesh_file, failed_dir)
        
        if should_move:
            # Create destination path preserving relative directory structure
            rel_path = mesh_file.relative_to(script_dir)
            dest_path = failed_dir / rel_path
            
            # Create necessary subdirectories
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move the file
            try:
                shutil.move(str(mesh_file), str(dest_path))
                print(f"Moved {mesh_file.name} to failed_meshes directory")
                failed_count += 1
            except Exception as e:
                print(f"Error moving file {mesh_file}: {e}")
        else:
            if process.returncode == 0:
                success_count += 1
            else:
                skipped_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success_count} meshes")
    print(f"Failed and moved: {failed_count} meshes")
    print(f"Skipped: {skipped_count} meshes")

if __name__ == "__main__":
    main()