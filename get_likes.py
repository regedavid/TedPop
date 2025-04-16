import pandas as pd
import requests
import time
import re
from urllib.parse import quote
import os
API_KEY = ""

df_main = pd.read_csv("kaggle_dataset/ted_main.csv")
df_url = pd.read_csv("kaggle_dataset/transcripts.csv")

df_main["like_count"] = None

stats_path = "youtube_stats_buffer.csv"

# Load buffer if it exists
if os.path.exists(stats_path):
    stats_buffer = pd.read_csv(stats_path)
else:
    stats_buffer = pd.DataFrame(columns=["video_name", "video_id", "like_count", "view_count", "comment_count"])

# Make a lookup for already parsed TED URLs
processed_urls = set(stats_buffer["video_name"])
stats_lookup = stats_buffer.set_index("video_name").to_dict(orient="index")

def update_buffer(ted_url, video_id):
    global buffer_df
    buffer_df = pd.concat([
        buffer_df,
        pd.DataFrame([{"url": ted_url, "video_id": video_id}])
    ], ignore_index=True)
    buffer_df.drop_duplicates(subset="url", inplace=True)
    buffer_df.to_csv(buffer_path, index=False)

def get_video_id_from_youtube_search(title, speaker):
    query = quote(f"{title} {speaker} TED")
    url = f"https://www.youtube.com/results?search_query={query}"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    res = requests.get(url, headers=headers)
    
    # Try to extract the first video ID (11 chars)
    match = re.search(r'"videoId":"([a-zA-Z0-9_-]{11})"', res.text)
    if match:
        return match.group(1)
    else:
        print("No video ID found in search results.")
        return None
    
def get_video_id(title):
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": title,
        "type": "video",
        "key": API_KEY,
        "maxResults": 1
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        items = response.json().get("items")
        if items:
            return items[0]["id"]["videoId"]
    return None

def get_video_stats(video_id):
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "statistics",
        "id": video_id,
        "key": API_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        stats = response.json().get("items")[0]["statistics"]
        return {
            "likeCount": stats.get("likeCount", "0"),
            "viewCount": stats.get("viewCount", "0"),
            "commentCount": stats.get("commentCount", "0")
        }
    return None

def extract_video_id(url):
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",      # youtube.com/watch?v=VIDEO_ID
        r"youtu\.be/([a-zA-Z0-9_-]{11})"  # youtu.be/VIDEO_ID
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

for i, row in df_main.iterrows():
    video_name = row["name"]
    if video_name in processed_urls:
        df_main.at[i, "like_count"] = stats_lookup[video_name]["like_count"]
        df_main.at[i, "view_count"] = stats_lookup[video_name]["view_count"]
        df_main.at[i, "comment_count"] = stats_lookup[video_name]["comment_count"]
        print(f"Skipping already processed: {video_name}")
        continue

    video_id = get_video_id_from_youtube_search(
    title=row["title"],
    speaker=row["main_speaker"]
    )
    if video_id:
        stats = get_video_stats(video_id)
        if stats is not None:
            df_main.at[i, "like_count"] = stats["likeCount"]
            df_main.at[i, "view_count"] = stats["viewCount"]
            df_main.at[i, "comment_count"] = stats["commentCount"]

            new_entry = {
                "video_name": video_name,
                "video_id": video_id,
                "like_count": stats["likeCount"],
                "view_count": stats["viewCount"],
                "comment_count": stats["commentCount"]
            }
            stats_buffer = pd.concat([stats_buffer, pd.DataFrame([new_entry])], ignore_index=True)
            stats_buffer.to_csv(stats_path, index=False)
            print(f"Processed {i+1}/{len(df_main)}: {video_id} - Likes: {df_main.at[i, 'like_count']}")
        else:
            print("Quota exceeded for stats printing")
            break
    else:
        print(f"Video ID not found for video: {video_name}")
        break
    
    time.sleep(0.1)

df_main.to_csv("ted_main_2.csv", index=False)
