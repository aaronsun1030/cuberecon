import cv2
import numpy as np
import os
import shutil

outliar_thresh = 2
cube_buffer = 100
min_sq_area = 100
mag_thresh = 1

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
        
        pts = get_points(frame2)
        pts = filter_points(pts)
        box = find_bounding_box(pts, frame2.shape)

        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        if np.average(mag) < mag_thresh:
            cur_label = "N"

        mag_ang = np.array([mag, ang])[:, box[2]:box[3], box[0]:box[1]]

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
                    frame_path = frames_path + os.sep + label + str(label_counts[label]) + ".npy"
                    np.save(frame_path, frame)

def filter_points(data, thresh=outliar_thresh):
    return data[np.all(data - np.mean(data, axis=0) < thresh * np.std(data, axis=0), axis=1).flatten()]

def get_points(frame):
    b, g, r = cv2.split(frame)

    #edge detection filter
    kernel = np.array([[0.0, -1.0, 0.0], 
                    [-1.0, 4.0, -1.0],
                    [0.0, -1.0, 0.0]])

    #filter the source image
    b_out = cv2.filter2D(b,-1,kernel)
    g_out = cv2.filter2D(g,-1,kernel)
    r_out = cv2.filter2D(r,-1,kernel)

    blur = cv2.GaussianBlur(cv2.cvtColor(cv2.merge((b_out, g_out, r_out)), cv2.COLOR_BGR2GRAY),(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    blurred = cv2.GaussianBlur(th3, (3, 3), 0)
    canny = cv2.Canny(blurred, 20, 40)
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(canny, kernel, iterations=2)
    (contours, hierarchy) = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hierarchy = hierarchy[0]
    candidates = []
    center = []

    for component in zip(contours, hierarchy):
        contour = component[0]
        curr_hierarchy = component[1]
        
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
        area = cv2.contourArea(contour)

        if len(approx) == 4 and len(cv2.convexHull(approx, returnPoints = False)) == 4 and curr_hierarchy[2] < 0 and area > min_sq_area:
            candidates.extend(approx)
            
    return np.array(candidates).reshape(-1, 2)

def find_bounding_box(points, dims, buffer=cube_buffer):
    min_x, min_y, max_x, max_y = dims[1], dims[0], 0, 0
    for point in points:
        min_x = min(min_x, point[0])
        max_x = max(max_x, point[0])
        min_y = min(min_y, point[1])
        max_y = max(max_y, point[1])
    min_x = max(min_x - buffer, 0)
    max_x = min(max_x + buffer, dims[1])
    min_y = max(min_y - buffer, 0)
    max_y = min(max_y + buffer, dims[0])
    return (min_x, max_x, min_y, max_y)

gen_dataset(mag_thresh)