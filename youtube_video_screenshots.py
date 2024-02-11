# pip install pytube
# pip install opencv-python


import cv2
import os
from pytube import YouTube

download_folder = "videos"
frame_folder = "frames"

if not os.path.exists(download_folder):
    os.makedirs(download_folder)

if not os.path.exists(frame_folder):
    os.makedirs(frame_folder)

import pdb
pdb.set_trace()

video_url = "https://www.youtube.com/watch?v=lHgxFfioaR4"

yt = YouTube(video_url)

#video_title = yt.title
#video_name = f"{yt.video_id}_{video_title}"
video_name = "temp_name"

def on_progress(stream, chunk, remaining):
    total_size = stream.filesize
    bytes_downloaded = total_size - remaining

    percentage = (bytes_downloaded / total_size) * 100
    print(f"\rDownloading {video_name}: {percentage:.2f}% complete", end="", flush=True)


video_stream = yt.streams.filter(res="720p").first() or yt.streams.get_highest_resolution()

video_path = video_stream.download(download_folder, on_progress_callback=on_progress)


def is_frame_different(frame1, frame2):
    diff = cv2.absdiff(frame1, frame2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_diff, 10, 255, cv2.THRESH_BINARY)
    num_non_zero_pixels = cv2.countNonZero(mask)
    return num_non_zero_pixels > (frame1.shape[0] * frame1.shape[1]) * 0.1

def save_frame(frame, video_name, frame_number):
    video_subfolder = os.path.join(download_folder, video_name)
    if not os.path.exists(video_subfolder):
        os.makedirs(video_subfolder)

    cv2.imwrite(os.path.join(video_subfolder, f"{video_name}_Frame_{frame_number}.jpg"), frame)

cap = cv2.VideoCapture(video_path)
frame_number = 1
previous_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if previous_frame is not None:
        if is_frame_different(previous_frame, frame):
            save_frame(frame, video_name, frame_number)
            frame_number += 1

    previous_frame = frame

cap.release()



