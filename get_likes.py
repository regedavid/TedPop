import pandas as pd
import requests
import time
import re
from urllib.parse import quote
import os
from pytube import YouTube
import json
import isodate
from youtube_dl import YoutubeDL

def get_ydl_opts(path):
        return {
            "format": "bestaudio/best",
            "outtmpl": f"{path}/%(id)s.%(ext)s",
            "ignoreerrors": True,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "320",
                }
            ],
        }

def ydl_opts(path):
    return {
        "outtmpl": f"{path}/%(id)s.%(ext)s",
    }

def parse_duration(iso_duration):
    try:
        duration = isodate.parse_duration(iso_duration)
        return duration.total_seconds()
    except:
        return None
    
def update_buffer(ted_url, video_id, buffer_path):
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
    video_ids = re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', res.text)
    video_ids = list(dict.fromkeys(video_ids))

    # Check first 5 video IDs
    for video_id in video_ids[:5]:
        if is_official_ted_video(video_id):
            print(f"✅ Found official TED video: {video_id}")
            return video_id
    
    print("❌ No official TED videos found among top 5 results.")
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

def is_official_ted_video(video_id):
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "snippet",
        "id": video_id,
        "key": API_KEY
    }
    res = requests.get(url, params=params)
    if res.status_code == 200:
        items = res.json().get("items", [])
        if items:
            channel_id = items[0]["snippet"]["channelId"]
            return channel_id == OFFICIAL_TED_CHANNEL_ID
    else:
        print(f"⚠️ Failed to check video {video_id} (status code {res.status_code})")
    return False

def download_youtube_video(video_id, output_path="downloads/"):
    url = f"https://www.youtube.com/watch?v={video_id}"
    with YoutubeDL(ydl_opts(output_path)) as ydl:
        try:
            #metadata = ydl.extract_info(url, download=False)
            ydl.download([url])
            print(f"✅ Downloaded video: {video_id}")
        except Exception as e:
            print(f"❌ Failed to download video {video_id}: {e}")

def get_youtube_metadata(video_id):
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "snippet,statistics,contentDetails,status",
        "id": video_id,
        "key": API_KEY
    }
    res = requests.get(url, params=params)
    if res.status_code == 200:
        items = res.json().get("items", [])
        if items:
            video = items[0]
            snippet = video.get("snippet", {})
            stats = video.get("statistics", {})
            details = video.get("contentDetails", {})
            status = video.get("status", {})

            return {
                "video_id": video_id,
                "title": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "publishedAt": snippet.get("publishedAt", ""),
                "duration": parse_duration(details.get("duration", "")),
                "definition": details.get("definition", ""),
                "caption": details.get("caption", ""),
                "licensedContent": details.get("licensedContent", False),
                "viewCount": stats.get("viewCount", 0),
                "likeCount": stats.get("likeCount", 0),
                "commentCount": stats.get("commentCount", 0),
                "channelTitle": snippet.get("channelTitle", ""),
                "tags": ", ".join(snippet.get("tags", [])),  # Join tags list into a string
                "categoryId": snippet.get("categoryId", "")
            }
    else:
        print(f"⚠️ Failed to fetch {video_id}, Status Code:", res.status_code)
    return None

def main_script(df_main, df_metadata, video_metadata_path):
    count_unfound = 0
    for i, row in df_main.iterrows():
        video_name = row["name"]
        if video_name in processed_urls:
            # df_main.at[i, "like_count"] = stats_lookup[video_name]["like_count"]
            # df_main.at[i, "view_count"] = stats_lookup[video_name]["view_count"]
            # df_main.at[i, "comment_count"] = stats_lookup[video_name]["comment_count"]
            print(f"Skipping already processed: {video_name}")
            continue

        video_id = get_video_id_from_youtube_search(
        title=row["title"],
        speaker=row["main_speaker"]
        )
        if video_id:
            md = get_youtube_metadata(video_id)
            #stats = get_video_stats(video_id)
            if md:
                # df_main.at[i, "like_count"] = stats["likeCount"]
                # df_main.at[i, "view_count"] = stats["viewCount"]
                # df_main.at[i, "comment_count"] = stats["commentCount"]

                # new_entry = {
                #     "video_name": video_name,
                #     "video_id": video_id,
                #     "like_count": stats["likeCount"],
                #     "view_count": stats["viewCount"],
                #     "comment_count": stats["commentCount"]
                # }
                md['old_title'] = video_name
                df_metadata = pd.concat([df_metadata, pd.DataFrame([md])], ignore_index=True)
                df_metadata.to_csv(video_metadata_path, index=False)
                print(f"Processed {i+1}/{len(df_main)}: {video_id}")

                download_youtube_video(video_id)
            else:
                print("Quota exceeded")
                break
        else:
            print(f" Official TED video ID not found for video: {video_name}. Number of unfound videos: {count_unfound}")
            count_unfound += 1
            continue

        time.sleep(0.1)

    df_main.to_csv("ted_main_refurbished.csv", index=False)

if __name__ == "__main__":
    API_KEY = ""
    OFFICIAL_TED_CHANNEL_ID = "UCAuUUnT6oDeKwE6v1NGQxug"

    columns = [
    "video_id", "title", "old_title", "description", "publishedAt", "duration",
    "definition", "caption", "licensedContent",
    "viewCount", "likeCount", "commentCount",
    "channelTitle", "tags", "categoryId"
]

    df_metadata = pd.DataFrame(columns=columns)

    df_main = pd.read_csv("kaggle_dataset/ted_main.csv")
    df_url = pd.read_csv("kaggle_dataset/transcripts.csv")

    df_main["like_count"] = None

    video_metadata_path = "video_metadata.csv"

    if os.path.exists(video_metadata_path):
        md_buffer = pd.read_csv(video_metadata_path)
    else:
        md_buffer = pd.DataFrame(columns=columns)

    processed_urls = set(md_buffer["old_title"])
    stats_lookup = md_buffer.set_index("old_title").to_dict(orient="index")

    main_script(df_main=df_main, df_metadata=df_metadata, video_metadata_path=video_metadata_path)
