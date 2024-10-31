import cv2
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_iris

# Đường dẫn đến thư mục chứa ảnh nha khoa
dental_image_dir = r"C:\Users\hungh\Downloads\Panoramic radiographs with periapical lesions Dataset\Panoramic radiographs with periapical lesions Dataset\Periapical Dataset\Periapical Lesions\Original JPG Images"

features = []
labels = []

# Lấy danh sách tất cả các ảnh nha khoa và chọn ngẫu nhiên 300 ảnh
dental_images = [img for img in os.listdir(dental_image_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
selected_dental_images = random.sample(dental_images, 300)


# Đọc và gán nhãn "0" cho ảnh nha khoa
for img_name in selected_dental_images:
    img_path = os.path.join(dental_image_dir, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    avg_gray = np.mean(img)
    contrast = np.std(img)

    feature_vector = [avg_gray, contrast]
    features.append(feature_vector)
    labels.append(0)  # Nhãn cho ảnh nha khoa là 0

# Tải dữ liệu IRIS và chuyển đổi thành ảnh
iris = load_iris()
iris_data = iris.data[:300]  # Giới hạn ở 150 mẫu đầu tiên

# Tạo ảnh từ dữ liệu IRIS và gán nhãn "1"
for sample in iris_data:
    img = np.reshape(sample, (2, 2))  # Chuyển đổi dữ liệu thành hình ảnh 2x2 (giả định)
    avg_gray = np.mean(img)
    contrast = np.std(img)

    feature_vector = [avg_gray, contrast]
    features.append(feature_vector)
    labels.append(1)  # Nhãn cho ảnh hoa là 1

# Chuyển đổi danh sách đặc trưng và nhãn sang mảng NumPy để huấn luyện mô hình
features = np.array(features)
labels = np.array(labels)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# CART classifier (Gini Index)
cart_clf = DecisionTreeClassifier(criterion='gini')
cart_clf.fit(X_train, y_train)
y_pred_cart = cart_clf.predict(X_test)

# ID3 classifier (Information Gain)
id3_clf = DecisionTreeClassifier(criterion='entropy')
id3_clf.fit(X_train, y_train)
y_pred_id3 = id3_clf.predict(X_test)

# In kết quả
print("CART (Gini Index) - Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_cart))
print(classification_report(y_test, y_pred_cart))

print("\nID3 (Information Gain) - Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_id3))
print(classification_report(y_test, y_pred_id3))