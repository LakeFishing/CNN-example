import cv2
from keras.models import load_model
import numpy as np

def cnn2_pre():
    model = load_model('cnn3.h5')

    img = cv2.imread('images\output.png', cv2.IMREAD_GRAYSCALE)

    res, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    img_2D = img.reshape(1, 28, 28).astype('float32')
    img_rorm = img_2D / 255
    img = img_rorm

    predictions = model.predict(img)

    print("CNN2：", predictions)
    c = np.argmax(predictions)
    return c

if __name__ == '__main__':
    cnn2_pre()