import tensorflow as tf
import numpy as np
import os
import cv2
from gen_flow_data import process_video

model = tf.keras.models.load_model("model")

video = "C:\\Users\\Aaron\\Pictures\\2020-12-23\\IMG_2764.mp4"

cap = cv2.VideoCapture(video)

encoding = ['N_', 'B_', 'Bp', 'D_', 'Dp', 'F_', 'Fp', 'L_', 'Lp', 'R_', 'Rp', 'U_', 'Up']

def display_flow(mag, ang):
    hsv = np.zeros((mag.shape[0], mag.shape[1], 3), dtype="uint8")
    hsv[...,1] = 255
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imshow('frame2',rgb)
    cv2.waitKey()

for x, y in process_video(video, 1):

    #x = np.array(tf.image.resize(np.moveaxis(x, 0, -1), size=(256, 256))).reshape((1, 256, 256, 2))
    ret, frame1 = cap.read()
    x = np.array(x).reshape(1, 256, 256, 2)
    predicted = model.predict(x)
    choice = np.argmax(predicted[0])
    #print([encoding[i] + ':' + str(predicted[0][i]) for i in range(len(encoding))])
    cv2.putText(frame1, encoding[choice] + ": " + str(predicted[0][choice]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
    cv2.imshow("aaa", frame1)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    #cv2.waitKey()
    #x = np.array(tf.image.resize(np.moveaxis(x, 0, -1), size=(256, 256))).reshape((1, 256, 256, 2))
    #display_flow(x[0, :, :, 0], x[0, :, :, 1])