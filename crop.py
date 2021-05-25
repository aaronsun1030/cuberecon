import cv2
import numpy as np
import math
import os

for subdir, dirs, files in os.walk('./data/crops'):
        for file in files:
            file_path = subdir + os.sep + file
            #for frame, label in process_video(file_path, mag_thresh):
            pass

cap = cv2.VideoCapture("./data/crops/IMG_2866.MOV")
while True:
    ret, frame = cap.read()

    b, g, r = cv2.split(frame)

    #edge detection filter
    kernel = np.array([[0.0, -1.0, 0.0], 
                    [-1.0, 4.0, -1.0],
                    [0.0, -1.0, 0.0]])

    #filter the source image
    b_out = cv2.filter2D(b,-1,kernel)
    g_out = cv2.filter2D(g,-1,kernel)
    r_out = cv2.filter2D(r,-1,kernel)

    #blur = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) ,(5,5),0)
    blur = cv2.GaussianBlur(cv2.cvtColor(cv2.merge((b_out, g_out, r_out)), cv2.COLOR_BGR2GRAY),(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.namedWindow("", cv2.WINDOW_NORMAL)
    th3 = cv2.morphologyEx(th3, cv2.MORPH_OPEN, np.ones((3, 3), dtype="uint8"))

    src = th3

    dst = cv2.Canny(th3, 50, 200, None, 3)
    cv2.namedWindow("", cv2.WINDOW_NORMAL)
    cv2.imshow("", dst)
    cv2.waitKey()
        
    # Copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    lines = cv2.HoughLines(dst, 1, np.pi / 180, 50, None, 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)


    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 70, None, 50, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

    #cv2.imshow("Source", src)
    #cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv2.namedWindow("Detected Lines (in red) - Probabilistic Line Transform", cv2.WINDOW_NORMAL)
    cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    cv2.waitKey()