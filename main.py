import cv2
import matplotlib.pyplot as plt

import numpy as np
import requests

url = 'https://images.unsplash.com/photo-1747669694605-cd90e9709beb?q=80&w=2080&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D'
response = requests.get(url)
image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 一画面に複数の画像を表示するための設定
plt.figure(figsize=(15, 10))

# オリジナル画像を表示
plt.subplot(1, 4, 1)
plt.imshow(img_gray, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# SIFT特徴検出
sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(img_gray, None)
img_kp = cv2.drawKeypoints(img_gray, kp, None, 
                           color=(0, 0, 255),
                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.subplot(1, 4, 2)
plt.imshow(img_kp)
plt.title('SIFT Keypoints')
plt.axis('off')

# ORB特徴検出
orb = cv2.ORB_create(nfeatures=1000)
kp, des = orb.detectAndCompute(img_gray, None)
img_kp = cv2.drawKeypoints(img_gray, kp, None, 
                           color=(0, 0, 255),
                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.subplot(1, 4, 3)
plt.imshow(img_kp)
plt.title('ORB Keypoints')
plt.axis('off')

# SURF特徴検出
# SURFは特許の関係で、OpenCVの標準モジュールには含まれていない
# そのため、opencv-contrib-pythonをインストールする必要がある
# SURFはSIFTよりも高速で、スケール不変性を持つ特徴点検出器
# ただし、SURFは特許の関係で商用利用にはライセンスが必要

# AKAZE特徴検出
akaze = cv2.AKAZE_create()
kp, des = akaze.detectAndCompute(img_gray, None)
img_kp = cv2.drawKeypoints(img_gray, kp, None, 
                           color=(0, 0, 255),
                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.subplot(1, 4, 4)
plt.imshow(img_kp)
plt.title('AKAZE Keypoints')
plt.axis('off')

plt.show()


