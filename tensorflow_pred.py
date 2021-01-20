import tensorflow as tf
import numpy as np
import os
import cv2
import random
import tensorboard
from datetime import datetime

def display_flow(mag, ang):
    hsv = np.zeros((mag.shape[0], mag.shape[1], 3), dtype="uint8")
    hsv[...,1] = 255
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imshow('frame2',rgb)
    cv2.waitKey()

encoding = ['D', 'Dp', 'N', 'R', 'Rp', 'U', 'Up']

def batch(split):
    all_dirs = list(os.walk("frame_data/"))[0][2]
    random.shuffle(all_dirs)
    if split == "val":
        all_dirs = all_dirs[len(all_dirs) * 4 // 5:]
    else:
        all_dirs = all_dirs[:len(all_dirs) * 4 // 5]
    X = []
    y = []
    while True:
        for file in all_dirs:
            if file.endswith(".npy"):
                X.append(tf.image.resize(np.swapaxes(np.load("frame_data/" + file), 0, 2), size=(256, 256)))
                y.append(encoding.index(file[0] + ("p" if file[1] == "p" else "")))
                if len(y) == 8:
                    X = np.array(X, dtype="uint8")
                    y = tf.keras.utils.to_categorical(y, num_classes=len(encoding))
                    yield X, y
                    X, y = [], []

'''model = tf.keras.applications.InceptionV3(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(256, 256, 2),
    pooling=None,
    classes=len(encoding),
    classifier_activation="softmax",
)
# we can see if it works with this
#model.summary()

# Define the Keras TensorBoard callback.

logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

model = tf.keras.models.load_model("model")
model.compile(optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'], loss=tf.keras.losses.CategoricalCrossentropy())
model.fit(x=batch("train"), epochs=20, steps_per_epoch=104, validation_data=batch("val"), validation_steps=26, callbacks=[tensorboard_callback])
model.save("model")'''

model = tf.keras.models.load_model("model")
for x, y in batch("val"):    
    predicted = model.predict(x)
    result = np.absolute(y-predicted)
    for i in range(len(x)):
        if np.argmax(predicted[i]) != np.argmax(y[i]):
            mag, ang = x[i, :, :, 0], x[i, :, :, 1]
            print("displaying image")
            print("prediction:", encoding[np.argmax(predicted[i])], "expected:", encoding[np.argmax(y[i])])
            display_flow(mag, ang)
        else:
            print("got it right!")