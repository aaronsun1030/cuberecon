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
buffer_size = 10

def label(vid, relabel=False):
    filename = vid.name
    if relabel:
        old_labels = list(np.load(str(labels_dir) + '/' + filename + '.npy'))
    cap = cv2.VideoCapture(str(video_dir) + '/' + filename)
    labels = []

    cv2.namedWindow("Display frame", cv2.WINDOW_NORMAL)
    
    frames = []
    while (frame := cap.read()[1]) is not None:
        frames.append(frame)
    frame_index = 0
    while frame_index < len(frames):
        frame = frames[frame_index].copy()
        write_moves(frame, labels, buffer_size)
        write_moves(frame, old_labels[:frame_index], buffer_size, below=True)
        cv2.imshow("Display frame", frame)
        c = None
        backspace = False
        while c not in keymapping:
            c = chr(cv2.waitKey(0))
            if c == '\b':
                labels.pop()
                frame_index -= 1
                backspace = True
                break
            elif c == chr(27):
                cap.release()
                return 
        if not backspace:
            if relabel and keymapping[c] != old_labels[frame_index]:
                cv2.putText(frame, "Confirm relabel?", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("Display frame", frame)
                if c == chr(cv2.waitKey(0)):
                    labels.append(keymapping[c])
                else:
                    labels.append(old_labels[frame_index])
            else:
                labels.append(keymapping[c])
            frame_index += 1
        
    np.save(str(labels_dir) + '/' + filename, np.array(labels))
    cap.release()

def write_moves(frame, labels, buffer_size, below=False):
    if len(labels) >= buffer_size:
        labels = labels[-buffer_size:]
    y = 100 if below else 50
    cv2.putText(frame, str(labels), (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

#label('IMG_2777.MOV')

def already_labeled(vid):
    return Path(str(labels_dir) + '/' + vid.name + '.npy').exists()

for vid in video_dir.iterdir():
    if str(vid)[-3:].lower() in ("mov", "mp4"):
        label(vid, already_labeled(vid))