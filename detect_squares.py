import cv2
import numpy as np
import math
import sklearn.cluster
import scipy.stats
# https://programmer.help/blogs/rubik-cube-recognition-using-opencv-edge-and-position-recognition.html

# Hyperparameters
outliar_thresh = 3
move_detect_thresh = 1.6
move_recog_thresh = 1.6
k = 4
kmeans = sklearn.cluster.KMeans(n_clusters=k)

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

        if len(approx) == 4 and curr_hierarchy[2] < 0 and 1400 < area < 5000:
            # compute the center of the contour
            candidates.extend(approx)
            M = cv2.moments(contour)
            if M["m00"]:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                center.append([[cX, cY]])
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
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[np.all(s < thresh, axis=2).flatten()]
    
def check_move(var1, var2, weight=move_detect_thresh):
    #print(var1, var2)
    return np.min(var1) * weight < np.min(var2)

def get_move(prev, corners):
    mean = np.mean(corners, axis=0)
    var = np.linalg.norm(np.var(corners, axis=0))
    kmean_out = kmeans.fit(prev.reshape(-1, 2))
    clusters = []
    for label in range(k):
        group = prev[kmean_out.labels_ == label]
        if check_move(np.var(group, axis=0), var, weight=move_recog_thresh):
            clusters.append(kmean_out.cluster_centers_[label])
    return kmean_out.cluster_centers_


cap = cv2.VideoCapture('./IMG_2765_Trim.mp4')

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

 # Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = start = filter_points(get_corners(old_gray))
U, S, VT = np.linalg.svd(p0.reshape(-1, 2))

while 1:
    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    init_var = np.var(p0, axis=0)

    while not check_move(np.var(p0, axis=0), init_var):
        #print(np.linalg.norm(p0.reshape(-1, 2) @ pca))
        print(scipy.stats.skew(p0, axis=0), scipy.stats.kurtosis(p0, axis=0))
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(frame,mask)
        
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.imshow('frame',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        cv2.waitKey()

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
        p0 = filter_points(p0)
    
    corners = filter_points(get_corners(frame_gray))
    get_move(p0, corners)
    p0 = corners
    break


    

cv2.destroyAllWindows()
cap.release()