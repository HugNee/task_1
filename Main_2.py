import os
import time
import numpy as np
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load mô hình MobileNetV2 đã được huấn luyện sẵn
model = MobileNetV2(weights="imagenet")


# Hàm để tiền xử lý ảnh đầu vào cho MobileNetV2
def preprocess_image(image, target_size=(224, 224)):
    image = cv2.resize(image, target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image


# Hàm để dự đoán nhãn cho ảnh
def predict_label(image, model):
    processed_image = preprocess_image(image)
    preds = model.predict(processed_image)
    label = decode_predictions(preds, top=1)[0][0][1]  # Lấy tên nhãn dự đoán cao nhất
    return label


# Hàm để load và dự đoán các ảnh từ một thư mục
def load_and_predict(folder, model):
    images = []
    true_labels = []
    predicted_labels = []

    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):  # Chỉ chọn file .jpg
            img_path = os.path.join(folder, filename)
            label = filename.split('_')[0]  # Lấy tên nhãn thật từ tên file
            image = cv2.imread(img_path)

            if image is not None:
                images.append(image)
                true_labels.append(label)

                # Dự đoán nhãn cho ảnh
                predicted_label = predict_label(image, model)
                predicted_labels.append(predicted_label)
            else:
                print(f"Could not read file: {img_path}")
        else:
            print(f"Skipping non-JPG file: {filename}")

    return true_labels, predicted_labels


# Hàm đánh giá mô hình: tính accuracy, precision, recall, và thời gian chạy
def evaluate_model(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')

    return accuracy, precision, recall


# Đường dẫn tới thư mục chứa ảnh mới cần phân loại
folder_path = r"C:\IMG"
# Thực hiện dự đoán và tính thời gian chạy
start_time = time.time()
true_labels, predicted_labels = load_and_predict(folder_path, model)
execution_time = time.time() - start_time

# Tính toán các độ đo
accuracy, precision, recall = evaluate_model(true_labels, predicted_labels)

# Hiển thị kết quả
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"Execution Time: {execution_time:.2f} seconds")
