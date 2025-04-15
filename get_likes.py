import pandas as pd
import requests
import time
import re

API_KEY = "AIzaSyD9JVFS9ZBKNpgjQ2F1q4dkAw63StFCltU"

df_main = pd.read_csv("kaggle_dataset/ted_main.csv")
df_url = pd.read_csv("kaggle_dataset/transcripts.csv")

df_main["like_count"] = None

def extract_youtube_id_from_ted(ted_url):
    try:
        res = requests.get(ted_url)
        if res.status_code == 200:
            html = res.text
            match = re.search(r'youtube\.com/embed/([a-zA-Z0-9_-]{11})', html)
            if match:
                return match.group(1)
    except Exception as e:
        print(f"Error fetching {ted_url}: {e}")
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
    return {"likeCount": "0", "viewCount": "0", "commentCount": "0"}

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

for i, row in df_url.iterrows():
    ted_url = row["url"]
    video_id = extract_youtube_id_from_ted(ted_url)
    if video_id:
        stats = get_video_stats(video_id)
        df_main.at[i, "like_count"] = stats["likeCount"]
        df_main.at[i, "view_count"] = stats["viewCount"]
        df_main.at[i, "comment_count"] = stats["commentCount"]
        print(f"Processed {i+1}/{len(df_main)}: {video_id} - Likes: {df_main.at[i, 'like_count']}")
    else:
        print(f"Video ID not found for video: {ted_url}")
        df_main.at[i, "like_count"] = "0"
        df_main.at[i, "view_count"] = "0"
        df_main.at[i, "comment_count"] = "0"
    
    time.sleep(0.5)

df_main.to_csv("ted_main_2.csv", index=False)
