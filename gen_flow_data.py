import cv2
import numpy as np
import os
import shutil

def process_video(file_path, mag_thresh):

    flow_frames, frame_labels = [], []
    
    label = os.path.basename(file_path)[0:2].replace(' ', '')

    cap = cv2.VideoCapture(file_path)

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    while frame2 is not None:
        cur_label = label

        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        if np.average(mag) < mag_thresh:
            cur_label = "N"
        mag_ang = np.array([mag, ang])

        flow_frames.append(mag_ang)
        frame_labels.append(cur_label)

        prvs = next
        ret, frame2 = cap.read()

    cap.release()
    return flow_frames, frame_labels

def gen_dataset(mag_thresh = 0, frames_path = "./frame_data", videos_path = "./data"):

    label_counts = {}
    if os.path.isdir(frames_path):
        shutil.rmtree(frames_path)
    frames, frame_labels = [], []
    os.mkdir(frames_path)
    for subdir, dirs, files in os.walk(videos_path):
        for file in files:
            if file.endswith(".mov"):
                file_path = subdir + os.sep + file
                print(file_path)
                video_frames, video_labels = process_video(file_path, mag_thresh)
                for frame, label in zip(video_frames, video_labels):
                    label_counts[label] = label_counts.get(label, 0) + 1
                    if not os.path.exists(frames_path + os.sep + label):
                        os.mkdir(frames_path + os.sep + label)
                    frame_path = frames_path + os.sep + label + os.sep + label + str(label_counts[label]) + ".npy"
                    np.save(frame_path, frame)
gen_dataset(1)