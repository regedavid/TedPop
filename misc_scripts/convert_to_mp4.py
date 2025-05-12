import os
import subprocess
import shutil

# Path to the folder with MKV filescs
folder_path = os.path.join(os.getcwd(), 'downloads')
archive_folder = os.path.join(os.getcwd(), 'archive')

input_extensions = ('.mkv', '.webm')
# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(input_extensions):
        input_path = os.path.join(folder_path, filename)
        output_filename = os.path.splitext(filename)[0] + '.mp4'
        output_path = os.path.join(folder_path, output_filename)
        
        # FFmpeg command to convert
        command = ['ffmpeg', '-i', input_path, '-codec', 'copy', output_path]
        print(f'Converting: {filename}')
        
        # Run command
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        if result.returncode == 0:
            shutil.move(input_path, os.path.join(archive_folder, filename))
            print(f'Moved original MKV to archive: {filename}')
        else:
            print(f'Failed to convert: {filename}')

print("All conversions completed.")