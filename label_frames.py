import cv2
import numpy as np
from pathlib import Path

keymapping = {
    'Q':'U_',
    'W':'R_',
    'E':'F_',
    'R':'L_',
    'T':'B_',
    'Y':'D_',

    'A':'Up',
    'S':'Rp',
    'D':'Fp',
    'F':'Lp',
    'G':'Bp',
    'H':'Dp',

    'Z':'z_',
    'X':'x_',
    'C':'y_',
    'V':'zp',
    'B':'xp',
    'N':'yp',

    'q':'u_',
    'w':'r_',
    'e':'f_',
    'r':'l_',
    't':'b_',
    'y':'d_',

    'a':'up',
    's':'rp',
    'd':'fp',
    'f':'lp',
    'g':'bp',
    'h':'dp',

    'U':'M_',
    'I':'S_',
    'O':'E_',
    'J':'Mp',
    'K':'Sp',
    'L':'Ep',
    
    '1':'N_', # no move
    '2':'2_',
    '3':'3_',
    '4':'4_',
    '5':'5_',
    '6':'6_',
    '7':'7_',
    '8':'8_'
}

labels_dir = Path('./labels')
video_dir = Path('./data/B_')

def label(vid):
    filename = vid.name
    cap = cv2.VideoCapture(str(video_dir) + '/' + filename)
    labels = []
    prev_frames = []

    cv2.namedWindow("Display frame", cv2.WINDOW_NORMAL)

    while (frame := cap.read()[1]) is not None:
        clean = frame.copy()
        prev_frames.append(clean)
        if len(prev_frames) >= 10:
            prev_frames.pop(0)
        if len(labels) >= 5:
            cv2.putText(frame, str(labels[-5:]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("Display frame", frame)
        c = None
        while c not in keymapping:
            c = chr(cv2.waitKey(0))
            if c == '\b':
                backspace(prev_frames, labels)
                frame = clean.copy()
                if len(labels) >= 5:
                    cv2.putText(frame, str(labels[-5:]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow("Display frame", frame)
        labels.append(keymapping[c])
    np.save(str(labels_dir) + '/' + filename, np.array(labels))
    cap.release()

def backspace(prev_frames, labels):
    if len(prev_frames) == 0:
        return None
    labels.pop()
    clean = prev_frames[-1]
    frame = clean.copy()
    if len(labels) >= 5:
        cv2.putText(frame, str(labels[-5:]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("Display frame", frame)
    c = None
    while c not in keymapping:
        c = chr(cv2.waitKey(0))
        if c == '\b':
            backspace(prev_frames[:-1], labels)
            frame = clean.copy()
            if len(labels) >= 5:
                cv2.putText(frame, str(labels[-5:]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow("Display frame", frame)
    labels.append(keymapping[c])

#label('IMG_2777.MOV')

def already_labeled(vid):
    return Path(str(labels_dir) + vid.name).exists()

for vid in video_dir.iterdir():
    if str(vid)[-3:].lower() in ("mov", "mp4") and not already_labeled(vid):
        label(vid)