import tensorflow as tf
import numpy as np
import os
import cv2

#[X, y]
encoding = ['D', 'Dp', 'N', 'R', 'Rp', 'U', 'Up']
X = []
y = []

for subdir, dirs, files in os.walk("frame_data/"):
    for file in files:
        if file.endswith(".npy"):
            X.append(tf.image.resize(np.load(subdir + os.sep + file), size=(256, 256)))
            y.append(encoding.index(subdir[subdir.index(os.sep) + 1:]))

y = tf.keras.utils.to_categorical(y, num_classes=len(encoding))

model = tf.keras.applications.InceptionV3(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(256, 256, 2),
    pooling=None,
    classes=7,
    classifier_activation="softmax",
)
# we can see if it works with this
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam, loss=tf.keras.losses.CategoricalCrossentropy)
model.fit(x=X, y=y, batch_size=8, epochs=1, validiation_split=0.2)
