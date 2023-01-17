# import library
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.utils import np_utils

# 載入 MNIST 資料庫，訓練集60000張，測試集10000張，共10類
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 建立簡單的線性執行的模型
model = Sequential()

# 將訓練集的標籤進行獨熱編碼
y_TrainOneHot = np_utils.to_categorical(y_train)
y_TestOneHot = np_utils.to_categorical(y_test)

# 將訓練集的輸入資料轉為四維矩陣
x_Train4D = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
x_Test4D = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# 將色彩範圍從 0~255 正規化成 0~1
x_Train_norm = x_Train4D / 255
x_Test_norm = x_Test4D / 255

# 建立 16 個 filter
# 每一個 filter 大小為 5x5
# 讓影像大小不變
model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same',
          input_shape=(28, 28, 1), activation='relu'))
# 16 個 28x28 的影像縮小為 16 個 14x14 的影像
model.add(MaxPooling2D(pool_size=(2, 2)))
# 36 個 14x14 的影像
model.add(Conv2D(filters=36, kernel_size=(5, 5), padding='same',
                 activation='relu'))
# 36 個 14x14 的影像縮小為 36 個 7x7 的影像
model.add(MaxPooling2D(pool_size=(2, 2)))
# Dropout 避免 overfitting
model.add(Dropout(0.25))
# 將 36 個 7x7 的影像，轉換為一維向量，也就是 36x7x7=1764
model.add(Flatten())
# 隱藏層，共 128 個神經元
model.add(Dense(128, activation='relu'))
# 每次訓練迭代時，會隨機在神經網路中放棄 50% 的神經元，避免 overfitting
model.add(Dropout(0.5))
# 輸出層，共 10 個神經元，對應 0~9 共 10 個數字
# 使用 softmax 將神經元的輸出轉換為每一個數字預測的機率
model.add(Dense(10, activation='softmax'))
# 選擇 loss function, optimizer
# categorical_crossentropy 適合分多類
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# validation_split: 80% 為訓練資料, 20% 驗證資料
# batch_size: 每一批次 300 筆資料
train_history = model.fit(x=x_Train_norm,
                          y=y_TrainOneHot, validation_split=0.2,
                          epochs=20, batch_size=300, verbose=2)

# 顯示分數
scores = model.evaluate(x_Test_norm, y_TestOneHot)
print(scores[1] * 100)

# 儲存模型
model.save('cnn3.h5')  # 99.36
