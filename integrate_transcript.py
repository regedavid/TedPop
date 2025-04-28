import pandas as pd
import os

def integrate_transcript(transcript_path, main_path, metadata_path, output_path):
    # Load the transcript and buffer data
    transcript_df = pd.read_csv(transcript_path)
    main_df = pd.read_csv(main_path)
    metadata_df = pd.read_csv(metadata_path)

    # Create a dictionary for quick lookup of transcripts by URL
    transcript_dict = dict(zip(transcript_df['url'], transcript_df['transcript']))

    # Initialize an empty list to store transcripts
    transcripts = {}

    for url in main_df['url']:
        transcript = transcript_dict.get(url, None)
        if transcript:
            transcripts[main_df.loc[main_df['url'] == url, 'name'].values[0]] = transcript

    metadata_df['transcript'] = None
    for i, row in metadata_df.iterrows():
        video_name = row['old_title']
        if video_name in transcripts:
            metadata_df.at[i, 'transcript'] = transcripts[video_name]
        else:
            metadata_df.at[i, 'transcript'] = None


    metadata_df.to_csv(output_path, index=False)

integrate_transcript(
    transcript_path="kaggle_dataset/transcripts.csv",
    main_path="kaggle_dataset/ted_main.csv",
    metadata_path="kaggle_dataset/video_metadata.csv",
    output_path="kaggle_dataset/ted_main_refurbished.csv"
)