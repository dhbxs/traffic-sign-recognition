import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image

import numpy
from keras.models import load_model

model = load_model('traffic_classifier.h5')

classes = {1: '限速20km/h',
           2: '限速30km/h',
           3: '限速50km/h',
           4: '限速60km/h',
           5: '限速70km/h',
           6: '限速80km/h',
           7: '限速结束 (80km/h)',
           8: '限速100km/h',
           9: '限速120km/h',
           10: '禁止超车',
           11: '禁止超过3.5吨的车辆通过',
           12: '交叉路口的通行权',
           13: '优先通行',
           14: '让行',
           15: '停',
           16: '禁止车辆',
           17: '超过3.5吨禁止',
           18: '不得进入',
           19: '一般注意事项',
           20: '左转危险',
           21: '右转危险',
           22: '双曲线',
           23: '道路崎岖',
           24: '道路湿滑',
           25: '右边的路变窄',
           26: '修路',
           27: '注意红绿灯',
           28: '注意行人',
           29: '注意儿童穿行',
           30: '注意自行车穿行',
           31: '当心冰雪',
           32: '野生动物穿行',
           33: '最高速度+通过限制',
           34: '右转车道',
           35: '左转车道',
           36: '直行车道',
           37: '直行或右行道',
           38: '直行或左行道',
           39: '靠右行驶',
           40: '靠左行驶',
           41: '掉头',
           42: '无路可走',
           43: '禁止超过3.5吨的车辆通过'}

# 初始化窗口
top = tk.Tk()
top.geometry('800x600')
top.title('交通标志识别')
top.configure(background='#ffffff')

label = Label(top, background='#ffffff', font=('Microsoft YaHei', 15, 'bold'))
sign_image = Label(top)


def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.convert('RGB')
    image = image.resize((30, 30))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    print(image.shape)
    pred = model.predict_classes([image])[0]
    sign = classes[pred + 1]
    print(sign)
    label.configure(foreground='#6AAFE6', text=sign)


def show_classify_button(file_path):
    classify_b = Button(top, text="识别", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#6AAFE6', foreground='#6AAFE6', font=('Microsoft YaHei', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        # uploaded = uploaded.convert('RGB')
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass


upload = Button(top, text="上传图像", command=upload_image, padx=10, pady=5)
upload.configure(background='#6AAFE6', foreground='#6AAFE6', font=('Microsoft YaHei', 10, 'bold'))

upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="交通标志识别", pady=20, font=('Microsoft YaHei', 20, 'bold'))
heading.configure(background='#ffffff', foreground='#6AAFE6')
heading.pack()
top.mainloop()
