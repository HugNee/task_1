import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh từ máy tính
def read_image(image_path):
    image = cv2.imread(r"C:\Users\hungh\Downloads\anh_1.jpg", cv2.IMREAD_GRAYSCALE)
    return image

# Tăng độ tương phản (tăng gamma)
def increase_contrast(image, alpha=1.5, beta=0):
    # alpha > 1 để tăng độ tương phản, beta dùng để điều chỉnh độ sáng
    contrasted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return contrasted_image

# Tạo ảnh âm tính
def negative_image(image):
    negative = cv2.bitwise_not(image)
    return negative

# Biến đổi log
def log_transform(image):
    c = 255 / np.log(1 + np.max(image))
    log_image = c * (np.log(1 + image))
    log_image = np.array(log_image, dtype=np.uint8)
    return log_image

# Cân bằng histogram
def equalize_histogram(image):
    equalized_image = cv2.equalizeHist(image)
    return equalized_image

# Hiển thị ảnh gốc và ảnh đã chỉnh sửa
def show_images(original, transformed, title_transformed):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Ảnh gốc')

    plt.subplot(1, 2, 2)
    plt.imshow(transformed, cmap='gray')
    plt.title(title_transformed)

    plt.show()

# Chương trình chính để chỉnh sửa ảnh
def main(image_path):
    # Đọc ảnh từ đường dẫn
    image = read_image(image_path)

    # Thực hiện các tác vụ
    contrasted_image = increase_contrast(image)
    negative = negative_image(image)
    log_transformed = log_transform(image)
    hist_equalized = equalize_histogram(image)

    # Hiển thị ảnh
    show_images(image, contrasted_image, "Tăng độ tương phản")
    show_images(image, negative, "Ảnh âm tính")
    show_images(image, log_transformed, "Biến đổi log")
    show_images(image, hist_equalized, "Cân bằng histogram")

# Đường dẫn đến ảnh của bạn
image_path = "path_to_your_image.jpg"  # Thay đổi thành đường dẫn của ảnh bạn muốn xử lý
main(image_path)
