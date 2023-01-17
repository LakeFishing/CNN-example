import tkinter as tk
from PIL import ImageGrab
import cv2
from CNN_pre import cnn_pre
from CNN2_pre import cnn2_pre

picture = "images\output.png"

choosecolor = "black"
result = "None"

root = tk.Tk()
root.title('人工智慧')
root.geometry('+200+200')

def paint(event):
    x1, y1 = (event.x, event.y)
    x2, y2 = (event.x + 10, event.y + 10)
    canvas.create_oval(x1, y1, x2, y2, fill = choosecolor, outline = choosecolor)

canvas = tk.Canvas(root, width = 140, height = 140)
canvas.grid(row = 0, columnspan = 2)
canvas.bind("<B1-Motion>", paint)

def getter():
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + 140
    y1 = y + 140
    ImageGrab.grab().crop((x, y, x1, y1)).save(picture)
    img = cv2.imread(picture)
    img = cv2.resize(img, (28, 28))
    cv2.imwrite(picture,  img)
    result1 = cnn_pre()
    cnn.config(text = "CNN1：" + str(result1))
    result2 = cnn2_pre()
    cnn2.config(text = "CNN2：" + str(result2))

def clear():
    canvas.delete("all")
    cnn.config(text = "CNN1：" + result)
    cnn2.config(text = "CNN2：" + result)

btn1 = tk.Button(root, text = "確定", command = lambda: getter())
btn1.grid(row = 1, column = 0)

btn2 = tk.Button(root, text = "清除", command = lambda: clear())
btn2.grid(row = 1, column = 1)

cnn = tk.Label(root, text = "CNN1：" + str(result), font = 20)
cnn.grid(row = 2, columnspan = 2)

cnn2 = tk.Label(root, text = "CNN2：" + str(result), font = 20)
cnn2.grid(row = 3, columnspan = 2)

root.mainloop()