import os
import sys
# --------------------------------------------------
# This script creates symlinks for all audio files in a directory
# It is useful for creating a folder to use as dataset root
# this folder will have symlinks to audio files you have stored elsewhere
# Usage: python create_symlinks.py [SOURCE_DIR] [TARGET_DIR]
# for the SOURCE_DIR, use the absolute path!!!
# --------------------------------------------------


# Create symlinks for all audio files in a directory
def create_symlinks(source_dir, target_dir):
    # Ensure the target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Walk through the source directory
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(('.mp3', '.wav', '.flac')):  # add any other audio extensions if needed
                source_file_path = os.path.join(root, file)
                symlink_path = os.path.join(target_dir, file)
                
                # Check if a symlink or file by the same name already exists in the target directory
                counter = 1
                while os.path.exists(symlink_path):
                    file_name, ext = os.path.splitext(file)
                    symlink_path = os.path.join(target_dir, f"{file_name}_{counter}{ext}")
                    counter += 1

                os.symlink(source_file_path, symlink_path)
                print(f"Created symlink for {source_file_path} at {symlink_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: script_name.py [SOURCE_DIR] [TARGET_DIR]")
        sys.exit(1)

    source_dir = sys.argv[1]
    target_dir = sys.argv[2]
    create_symlinks(source_dir, target_dir)
