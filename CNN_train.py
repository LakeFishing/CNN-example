from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense
from keras.utils import np_utils

# 載入 MNIST 資料庫，訓練集60000張，測試集10000張，共10類
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 建立簡單的線性執行的模型
model = Sequential()

# 新增輸入層, 隱藏層有 256個輸出變數
model.add(Dense(units=256, input_dim=784,
          kernel_initializer='normal', activation='relu'))
# 新增輸出層
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))

# 選擇損失函數、優化方法及成效衡量方式
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 將訓練集的標籤進行獨熱編碼
y_TrainOneHot = np_utils.to_categorical(y_train)
y_TestOneHot = np_utils.to_categorical(y_test)

# 將訓練集的輸入資料轉為二維陣列
X_train_2D = X_train.reshape(60000, 28*28).astype('float32')
X_test_2D = X_test.reshape(10000, 28*28).astype('float32')

# 將色彩範圍從 0~255 正規化成 0~1
x_Train_norm = X_train_2D / 255
x_Test_norm = X_test_2D / 255

# 進行訓練
train_history = model.fit(x=x_Train_norm, y=y_TrainOneHot, validation_split=0.2, epochs=50, batch_size=800, verbose=2)

# 顯示分數
scores = model.evaluate(x_Test_norm, y_TestOneHot)
print(scores[1] * 100)

# 儲存模型
model.save('cnn1.h5')  # 98.0