import tensorflow as tf
import numpy as np
import os
import cv2
import random
import tensorboard
from datetime import datetime

#[X, y]
encoding = ['D', 'Dp', 'N', 'R', 'Rp', 'U', 'Up']
#print(list(os.walk("frame_data/")))
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
                    X = np.array(X)
                    y = tf.keras.utils.to_categorical(y, num_classes=len(encoding))
                    yield X, y
                    X, y = [], []

model = tf.keras.applications.InceptionV3(
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
model.save("model")