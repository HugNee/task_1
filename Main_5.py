import cv2
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skimage import io

# Bước 1: Đọc và tiền xử lý ảnh
image_path = r'C:\Users\hungh\Downloads\Sc_1.jpg'  # Đường dẫn đến ảnh vệ tinh trên máy tính
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_reshaped = image_rgb.reshape((-1, 3))

# Bước 2: Thực hiện phân cụm K-means
def kmeans_clustering(image_data, num_clusters=3):
    image_data = np.float32(image_data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(image_data, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    clustered_image = centers[labels.flatten()].reshape(image.shape)
    return clustered_image

# Bước 3: Thực hiện phân cụm Fuzzy C-means
def fuzzy_cmeans_clustering(image_data, num_clusters=3):
    data = np.float64(image_data.T)  # Chuyển đổi để phù hợp với FCM
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(data, num_clusters, 2, error=0.005, maxiter=1000)
    cluster_membership = np.argmax(u, axis=0)
    segmented_image = cntr[cluster_membership].astype(int).reshape(image.shape)
    return segmented_image

# Số lượng cụm mong muốn
num_clusters = 3

# Thực hiện phân cụm K-means và hiển thị kết quả
kmeans_result = kmeans_clustering(image_reshaped, num_clusters)
plt.subplot(1, 2, 1)
plt.title('K-means Clustering')
plt.imshow(kmeans_result)

# Thực hiện phân cụm Fuzzy C-means và hiển thị kết quả
fcm_result = fuzzy_cmeans_clustering(image_reshaped, num_clusters)
plt.subplot(1, 2, 2)
plt.title('Fuzzy C-means Clustering')
plt.imshow(fcm_result)

plt.show()
