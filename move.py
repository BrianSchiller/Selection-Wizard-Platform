import os
import shutil

src_dir = 'Output/Gen-Test_Def_20240725_12-52-07'
dst_dir = 'Output/Gen-Test_20240725_12-52-54'

for root, dirs, files in os.walk(src_dir):
        for dir_name in dirs:
            if dir_name.endswith('_processed') and "Conf" in dir_name:
                # Determine the path of the source and destination directories
                src_path = os.path.join(root, dir_name)
                # Extract the B_x_D_y part of the path
                relative_path = os.path.relpath(root, src_dir)
                dst_path = os.path.join(dst_dir, relative_path, dir_name)

                # Create destination path if it does not exist
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)

                # Copy the directory
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                print(f"Copied {src_path} to {dst_path}")