import os
import ffmpeg
import numpy as np

# Set your input/output directories
VIDEO_DIR = os.path.join(os.getcwd(), 'downloads/')      # folder with .mp4 files
AUDIO_DIR = os.path.join(os.getcwd(), 'audio_wav/')      # where .wav files will be saved # where features (.npy) will be saved


# Ensure output folders exist
os.makedirs(AUDIO_DIR, exist_ok=True)

def extract_audio_from_video(video_path, audio_path):
    try:
        ffmpeg.input(video_path).output(audio_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000').run(quiet=True, overwrite_output=True)
    except ffmpeg.Error as e:
        print(f"Failed to extract audio from {video_path}: {e}")

def process_all_videos():
    for filename in os.listdir(VIDEO_DIR):
        if filename.endswith('.mp4'):
            video_path = os.path.join(VIDEO_DIR, filename)
            audio_filename = filename.replace('.mp4', '.wav')
            audio_path = os.path.join(AUDIO_DIR, audio_filename)
            
            extract_audio_from_video(video_path, audio_path)
            
    

if __name__ == "__main__":
    process_all_videos()