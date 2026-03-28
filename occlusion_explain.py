import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random

# Load trained model
model = tf.keras.models.load_model("cnn_medical_model.h5")

# Folder containing pneumonia images
folder = "dataset/train/PNEUMONIA"

# Get all image names
images = os.listdir(folder)

# Randomly select an image
img_name = random.choice(images)

img_path = os.path.join(folder, img_name)

print("Selected image:", img_name)

# Read image
img = cv2.imread(img_path)

if img is None:
    print("Error loading image.")
    exit()

img = cv2.resize(img,(224,224))
img_norm = img/255.0

heatmap = np.zeros((224,224))

patch_size = 20

for y in range(0,224,patch_size):
    for x in range(0,224,patch_size):

        occluded = img_norm.copy()
        occluded[y:y+patch_size, x:x+patch_size] = 0

        input_img = np.expand_dims(occluded,axis=0)

        pred = model.predict(input_img)[0][0]

        heatmap[y:y+patch_size, x:x+patch_size] = pred

heatmap = (heatmap - heatmap.min())/(heatmap.max()-heatmap.min())

heatmap = np.uint8(255*heatmap)
heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)

superimposed = cv2.addWeighted(img,0.6,heatmap,0.4,0)

plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Occlusion Sensitivity Explanation")
plt.show()