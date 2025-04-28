# TedPop
What makes a presentation great?

# Dataset
Main dataset will be in dataset/ted_main_refurbished.csv with 2413 entries.
ted_main.csv dataset contained some (137) videos which are not from the official TED channel. These reuploads usually have very low engagement scores (views, likes, comments) and can skew the model estimations. Therefore, these videos have been removed.

The new dataset contains the following information about a specific TED video:
[video_id, title, old_title, description, publishedAt, duration, definition, caption, licensedContent, viewCount, likeCount, commentCount, channelTitle, tags, categoryId, transcript]

title :str - Name of the video as it appears at the current moment on Youtube
old_title :str - Name of the video as it appears in the old kaggle dataset. The videos' names were probably changed at some point.
publishedAt :str - Date of upload
duration :double - Duration of the video in seconds
definition :str - Quality of the video (Standard (SD) or High (HD) definition)
caption :bool - Whether captions are available or not
licensedContent :bool -
viewCount :int - Views at current point of time
likeCount :int - Likes at current point of time
commentCount :int - Only top level comments
channelTitle :str - Channel Title
tags :str - Tags for the video
categoryID :int - 
transcript :str - Transcript from the kaggle dataset


ted_main.csv contains some additional information about the videos which can be used for feature engineering: num_speakers, num_languages, ratings, speaker_occupation. These can easily be integrated in the main dataset.

# Audio/Video
Videos (mp4 format) for the 2413 entries have been downloaded on my personal machine. All videos take up roughly 180GB of space. I don't know if we're allowed to upload them all to the university servers (limit is 100GB). Even so, this limit is not strictly enforced so maybe we can get away for a week while doing trainings. 