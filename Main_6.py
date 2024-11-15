import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf
import tensorflow_hub as hub
import time
from sklearn.metrics import precision_score, recall_score
# Tải dữ liệu Oxford-IIIT Pet
dataset, info = tfds.load("oxford_iiit_pet", with_info=True, as_supervised=True)

train_data = dataset['train']
test_data = dataset['test']

# Thông tin tập dữ liệu
print(info)


# Định nghĩa các hàm tiền xử lý
IMG_SIZE = 128

def preprocess_image(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0  # Chuẩn hóa giá trị pixel
    label = tf.cast(label, tf.int32)  # Chuyển nhãn thành số nguyên
    return image, label

# Áp dụng tiền xử lý
train_data = train_data.map(preprocess_image).batch(32).shuffle(1000)
test_data = test_data.map(preprocess_image).batch(32)


# Khởi tạo mô hình CNN
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(37, activation='softmax')  # 37 lớp (loại thú cưng)
])

cnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
history = cnn_model.fit(train_data, validation_data=test_data, epochs=10)

# Đánh giá trên tập kiểm tra
loss, accuracy = cnn_model.evaluate(test_data)
print(f"Accuracy: {accuracy:.2f}")
# Lấy nhãn bounding box và chuyển đổi định dạng
def preprocess_bounding_box(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    label = tf.cast(label, tf.int32)
    return image, label

train_data = dataset['train'].map(preprocess_bounding_box).batch(32)
test_data = dataset['test'].map(preprocess_bounding_box).batch(32)


# Tải Faster R-CNN từ TensorFlow Hub
model = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/openimages_v4/inception_resnet_v2/1")

# Chạy phát hiện đối tượng trên một ảnh
for image, label in test_data.take(1):
    detections = model(image)

    # Hiển thị các kết quả
    print(detections)


# Đo thời gian chạy mô hình CNN
start_time = time.time()
cnn_predictions = cnn_model.predict(test_data)
cnn_time = time.time() - start_time

# Tính precision và recall
cnn_true_labels = []
cnn_pred_labels = []

for image, label in test_data.unbatch():
    cnn_true_labels.append(label.numpy())
    cnn_pred_labels.append(tf.argmax(cnn_predictions).numpy())

cnn_precision = precision_score(cnn_true_labels, cnn_pred_labels, average='macro')
cnn_recall = recall_score(cnn_true_labels, cnn_pred_labels, average='macro')

print(f"CNN: Time = {cnn_time:.2f}s, Precision = {cnn_precision:.2f}, Recall = {cnn_recall:.2f}")
