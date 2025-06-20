import numpy as np
import cv2
from matplotlib import pyplot as plt

#CODE PHẦN 1: HUẤN LUYỆN
img = cv2.imread('digits.png',0);
cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]
#print(cells[0][0])
cv2.imwrite('anh2.png',cells[9][34]);
#cv2.imshow("Number",cells[][34])

x = np.array(cells)
train = x[:,:50].reshape(-1,400).astype(np.float32)
test = x[:,50:100].reshape(-1,400).astype(np.float32)
k = np.arange(10)
#print(k)
train_labels = np.repeat(k,250) [:, np.newaxis]
test_labels = np.repeat(k,250) [:, np.newaxis]

knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE,train_labels)
kq1, kq2, kq3, kq4 = knn.findNearest(test,5)
_, result, _, _ = knn.findNearest(test, k=5)


index = int(input("Nhập chỉ số test (0 – 1249): "))

print("Nhãn thật:",test_labels[index])
print("Nhãn dự đoán:",kq2[index])


row = index // 50
col = 50 + (index % 50)
img_number = x[row][col]  # ảnh 20x20

# Hiển thị ảnh số đã chọn
img_resized = cv2.resize(img_number, (200, 200), interpolation=cv2.INTER_NEAREST)
cv2.imshow("{}".format(index), img_resized)

print("Nhấn Enter để thoát...")
while True:
    key = cv2.waitKey(0)
    if key == 13:
        break

cv2.destroyAllWindows()