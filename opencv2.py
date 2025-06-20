import numpy as np
import cv2

# PHẦN 1: HUẤN LUYỆN
img = cv2.imread('digits.png',0);
cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]
x = np.array(cells)

train = x[:, :50].reshape(-1, 400).astype(np.float32)
test = x[:, 50:100].reshape(-1, 400).astype(np.float32)


k = np.arange(10)
train_labels = np.repeat(k, 250)[:, np.newaxis]
test_labels = np.repeat(k, 250)[:, np.newaxis]

knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

ret, result, neighbours, dist = knn.findNearest(test, k=5)
matches = result == test_labels.astype(np.float32)
correct = np.count_nonzero(matches)
accuracy = correct * 100.0 / result.size
print("Độ chính xác trên tập kiểm thử: {:.2f}%".format(accuracy))

# PHẦN 2: NHẬP ẢNH VÀ TIỀN XỬ LÝ
def preprocess_image(image_path):
    img_input = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_input is None:
        raise ValueError("Không thể đọc ảnh!")

    img_eq = cv2.equalizeHist(img_input)
    blurred = cv2.GaussianBlur(img_eq, (3, 3), 0)

    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mean_value = np.mean(binary)
    if mean_value > 127:
        binary = 255 - binary  # Đảo ngược nền trắng, số đen

    contours, _ = cv2.findContours(255 - binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Không tìm thấy chữ số trong ảnh!")

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    roi = binary[y:y+h, x:x+w]

    # Làm ảnh vuông
    size = max(w, h)
    square = 255 * np.ones((size, size), dtype=np.uint8)
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = roi

    resized = cv2.resize(square, (20, 20), interpolation=cv2.INTER_AREA)
    return resized.reshape(1, 400).astype(np.float32), resized



# PHẦN 3: GOI CHƯƠNG TRINH CON
img_flatten, img_display = preprocess_image("anh9.png")
_, result, _, _ = knn.findNearest(img_flatten, k=5)


cv2.imshow("Result", cv2.resize(img_display, (200, 200), interpolation=cv2.INTER_NEAREST))
print("Số được nhận dạng là:", int(result[0][0]))
print("Nhấn ENTER để thoát...")

while True:
    if cv2.waitKey(0) == 13:
        break
cv2.destroyAllWindows()


