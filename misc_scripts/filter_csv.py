import os
import pandas as pd

# Config
csv_path = os.path.join(os.getcwd(), "tedpop/dataset/ted_main_refurbished.csv")             # original CSV
output_csv_path = os.path.join(os.getcwd(), "filtered_ted_refurbished.csv") # new cleaned CSV
downloads_folder = os.path.join(os.getcwd(), "downloads")
id_column = "video_id"
file_extension = '.mp4'

# Load CSV
df = pd.read_csv(csv_path)

# Filter rows based on file existence
def file_exists(talk_id):
    file_name = f"{talk_id}{file_extension}"
    return os.path.isfile(os.path.join(downloads_folder, file_name))

df_filtered = df[df[id_column].apply(file_exists)].reset_index(drop=True)

# Save new CSV
df_filtered.to_csv(output_csv_path, index=False)

print(f"Filtered CSV saved to {output_csv_path} — {len(df_filtered)} rows kept.")