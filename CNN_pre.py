import cv2
from keras.models import load_model
import numpy as np

def cnn_pre():
    model = load_model('cnn1.h5')

    # 顏色灰階
    img = cv2.imread('images\output.png', cv2.IMREAD_GRAYSCALE)

    # 從白底黑字轉成黑底白字
    res, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # 轉換編碼及格式
    img_2D = img.reshape(1, 28 * 28).astype('float32')
    img_norm = img_2D / 255
    img = img_norm

    # 進行辨識
    predictions = model.predict(img)
    
    # 返回預測結果
    print("CNN：", predictions)
    c = np.argmax(predictions)
    return c

if __name__ == '__main__':
    cnn_pre()