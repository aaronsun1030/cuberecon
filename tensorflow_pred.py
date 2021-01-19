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
    all_dirs = list(os.walk("frame_data/"))
    random.shuffle(all_dirs)
    if split == "val":
        all_dirs = all_dirs[len(all_dirs) * 4 // 5:]
    else:
        all_dirs = all_dirs[:len(all_dirs) * 4 // 5]
    while True:
        for subdir, dirs, files in all_dirs:
            for file in files:
                if file.endswith(".npy"):
                    yield (tf.reshape(tf.image.resize(np.swapaxes(np.load(subdir + '/' + file), 0, 2), size=(256, 256)), (1, 256, 256, 2)), 
                    tf.reshape(tf.one_hot(encoding.index(subdir[subdir.index('/') + 1:]), len(encoding)), (1, len(encoding))))

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

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy())
model.fit(x=batch("train"), epochs=5, steps_per_epoch=600, validation_data=batch("val"), validation_batch_size=100, validation_steps=5, callbacks=[tensorboard_callback])
model.save("model")