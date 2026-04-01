import tensorflow as tf
import numpy as np
import cv2
import os
import random

model = tf.keras.models.load_model("cnn_medical_model.h5")

folder = "dataset/train/PNEUMONIA"

images = os.listdir(folder)

img_name = random.choice(images)

img_path = os.path.join(folder, img_name)

print("Testing image:", img_name)

img = cv2.imread(img_path)
img = cv2.resize(img,(224,224))
img = img/255.0
img = np.expand_dims(img,axis=0)

prediction = model.predict(img)[0][0]

if prediction > 0.5:
    print("Prediction: PNEUMONIA")
else:
    print("Prediction: NORMAL")

print("Confidence:",prediction)