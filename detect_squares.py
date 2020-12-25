import cv2
import numpy as np
import sklearn.cluster
import scipy.stats
import os
# https://programmer.help/blogs/rubik-cube-recognition-using-opencv-edge-and-position-recognition.html


l = ['R (46).mov', 'Rp (8).mov', 'Dp (46).mov', 'D (23).mov', 'U (19).mov',
 'Rp (10).mov', 'D (60).mov', 'Rp (4).mov', 'U (38).mov', 'Rp (6).mov',
 'Dp (21).mov', 'Dp (4).mov,' 'Dp (24).mov', 'R (18).mov', 'D (59).mov',
 'U (1).mov', 'R (19).mov', 'Rp (5).mov', 'Up (5).mov', 'U (12).mov',
 'Dp (37).mov', 'D (16).mov', 'U (18).mov', 'U (37).mov', 'Up (41).mov',
 'Dp (25).mov', 'D (49).mov', 'R (7).mov', 'R (48).mov', 'Dp (31).mov',
 'D (5).mov', 'Up (20).mov']


# Hyperparameters
outliar_thresh = 2
n_frames = 5
k = 6
kmeans = sklearn.cluster.KMeans(n_clusters=k)
move_labels = {'U ': 0, 'Up': 1, 'D ': 2, 'Dp': 3, 'R ': 4, 'Rp': 5}

def get_corners(gray):
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
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

        if len(approx) == 4 and curr_hierarchy[2] < 0 and 1400 < area < 10000:
            candidates.extend(approx)
    """for i in range(len(candidates)):
        cv2.circle(gray, tuple(candidates[i][0]), 4, (0, 255, 0))
    for i in range(len(center)):
        cv2.circle(gray, tuple(center[i][0]), 4, (255, 255, 255))
    #cv2.drawContours(gray, center, -1, (255, 0, 255), 3)
    cv2.namedWindow("a", cv2.WINDOW_NORMAL)
    cv2.imshow("a", gray)
    cv2.waitKey()"""
    return np.array(candidates, dtype='float32')

def filter_points(data, thresh=outliar_thresh):
    return data[np.all(data - np.mean(data, axis=0) < thresh * np.std(data, axis=0), axis=2).flatten()]

def featurize(p0, p1):
    """Takes p0 and p1 and turns it into a kFRAME, which is the kmeans result of an array of [x, y, dx, dy] points."""
    non_outliar = np.all(p1 - np.mean(p1, axis=0) < outliar_thresh * np.std(p1, axis=0), axis=2).flatten()
    p0 = p0[non_outliar].reshape(-1, 2)
    p1 = p1[non_outliar].reshape(-1, 2)
    for p in [p0, p1]:
        for i in range(2):
            p[:, i] = scipy.stats.zscore(p[:, i])
    delta = p1 - p0
    frame = np.hstack((p0, delta))
    k_frame = kmeans.fit(frame)
    return sorted(k_frame.cluster_centers_, key=lambda x: x[0])


def get_data_point(cap):
    features = []

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = filter_points(get_corners(old_gray))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while ret:
        #print(np.linalg.norm(p0.reshape(-1, 2) @ pca))
        #print(scipy.stats.skew(p0, axis=0), scipy.stats.kurtosis(p0, axis=0))
        #kurtz = scipy.stats.skew(p0, axis=0)
        #kurtx.append(kurtz[0, 0])
        #kurty.append(kurtz[0, 1])

        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        features.append(featurize(p0, p1))

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel().astype(int)
            c,d = old.ravel().astype(int)
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(frame,mask)
            
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.imshow('frame',img)
        ka = cv2.waitKey(30) & 0xff
        if ka == 27:
            break
        cv2.waitKey()

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
        p0 = filter_points(p0) 

    #cv2.destroyAllWindows()
    cap.release()
    return np.mean(np.array(features), axis=0)
    return np.array(features)[np.linspace(0, len(features) - 1, n_frames, dtype = int)]

data = []
labels = []
file_name = []
for subdir, dirs, files in os.walk("./data"):
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file
        print(filepath)
        if file.endswith(".mov") and file in l:
            move = file[0:2]
            cap = cv2.VideoCapture(filepath)
            data.append(get_data_point(cap).flatten())
            labels.append(move_labels[move])
            file_name.append(file)

np.save("./data/data.npy", np.array(data))
np.save("./data/labels.npy", np.array(labels))
np.save("./data/filename.npy", np.array(file_name, dtype=object))

