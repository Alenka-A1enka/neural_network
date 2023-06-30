import PySimpleGUI as sg
import tkinter.messagebox as mb
import numpy as np, sys
np.random.seed(1)
from keras.datasets import mnist
import re
from PIL import Image, ImageDraw

layout = [
    [sg.Submit('Обучить нейросеть')],
    [sg.Output(size=(88, 20))],
    [sg.Submit('Распознать изображение')],
    [sg.Cancel()]
]
window = sg.Window('File Compare', layout)

weights_0_1 = [0]
weights_1_2 = [0]
current_test_image = 0
test_labels = []
test_images = []

while True:                            
    event, values = window.read()
    if event in (None, 'Exit', 'Cancel'):
        break

    if event == 'Обучить нейросеть':
        (x_train, y_train), (x_test, y_test) = mnist.load_data() #загрузка данных из библиотеки для тренировки и теста
        images, labels = (x_train[0:1000].reshape(1000,28*28) / 255, y_train[0:1000])
        one_hot_labels = np.zeros((len(labels),10))
        for i,l in enumerate(labels):
            one_hot_labels[i][l] = 1
        labels = one_hot_labels
        test_images = x_test.reshape(len(x_test),28*28) / 255
        test_labels = np.zeros((len(y_test),10))
        for i,l in enumerate(y_test):
            test_labels[i][1] = 1
        np.random.seed(1)

        #функции активации
        def relu(x):
            return (x >= 0) * x 
        def relu2deriv(output):
            return output >= 0

        # переменные
        alpha, iterations, hidden_size = (0.005, 30, 100)
        pixels_per_image, num_labels = (784, 10)

        #инициализация весов в диапазоне -0.1; 0.1
        weights_0_1 = 0.2*np.random.random((pixels_per_image,hidden_size))-0.1
        weights_1_2 = 0.2*np.random.random((hidden_size,num_labels)) - 0.1

        for j in range(iterations):
            correct_cnt = 0
            error = 0.0
            for i in range(len(images)):
                layer_0 = images[i:i+1] #сама картинка
                layer_1 = relu(np.dot(layer_0,weights_0_1)) #перемножение с весами первого слоя и обертка в функцию активации
                #маска против переобучения
                dropout_mask = np.random.randint(2,size=layer_1.shape)
                layer_1 *= dropout_mask * 2
                layer_2 = np.dot(layer_1,weights_1_2) #переменожение с весами второго слоя

                error += np.sum((labels[i:i+1] - layer_2) ** 2) #величина ошибки
                correct_cnt += int(np.argmax(layer_2) == np.argmax(labels[i:i+1])) #правильно ли распознаны числа

                layer_2_delta = (labels[i:i+1] - layer_2) #разница прогноза во втором слое
                layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1) #разница в первом слое 
                layer_1_delta *= dropout_mask #маска
                weights_1_2 += alpha * layer_1.T.dot(layer_2_delta) #изменение значений весов 
                weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)
            if(j % 1 == 0):
                print("\n"+ "Итерация:" + str(j) + " . Процент распознавания:" + str(correct_cnt/float(len(images))))

        msg = "Обучение закончено"
        mb.showinfo("Информация", msg)
    
            

    if event == 'Распознать изображение':
        layer_0 = test_images[current_test_image:current_test_image+1]
        layer_1 = relu(np.dot(layer_0, weights_0_1)) 
        layer_2 = np.dot(layer_1, weights_1_2)
        for i in range(1):
            #Создаем пустое изображение
            img = Image.new('RGB', (500, 500), (0,0,0))

            #Получаем объект рисования для нашей пикчи
            draw = ImageDraw.Draw(img)

            x = 10
            y = 10

            #Рисуем 10 квадратов с небольшими зазорами.
            for j in range(784):
                if j % 28 == 0 and j != 0:
                    y += 15
                    x = 10
                if layer_0[i][j] < 0.1:
                    draw.rectangle((x,y,x+10,y+10), fill='rgb(0,0,0)', outline=(0, 0, 0))
                elif layer_0[i][j] < 0.2:
                    draw.rectangle((x,y,x+10,y+10), fill='rgb(32,32,32)', outline=(0, 0, 0))
                elif layer_0[i][j] < 0.3:
                    draw.rectangle((x,y,x+10,y+10), fill='rgb(64,64,64)', outline=(0, 0, 0))
                elif layer_0[i][j] < 0.4:
                    draw.rectangle((x,y,x+10,y+10), fill='rgb(96,96,96)', outline=(0, 0, 0))
                elif layer_0[i][j] < 0.5:
                    draw.rectangle((x,y,x+10,y+10), fill='rgb(120,120,120)', outline=(0, 0, 0))
                elif layer_0[i][j] < 0.6:
                    draw.rectangle((x,y,x+10,y+10), fill='rgb(152,152,152)', outline=(0, 0, 0))
                elif layer_0[i][j] < 0.7:
                    draw.rectangle((x,y,x+10,y+10), fill='rgb(176,176,176)', outline=(0, 0, 0))
                elif layer_0[i][j] < 0.8:
                    draw.rectangle((x,y,x+10,y+10), fill='rgb(200,200,200)', outline=(0, 0, 0))
                elif layer_0[i][j] < 0.9:
                    draw.rectangle((x,y,x+10,y+10), fill='rgb(224,224,224)', outline=(0, 0, 0))
                else:
                    draw.rectangle((x,y,x+10,y+10), fill='rgb(255,255,255)', outline=(0, 0, 0))

                x += 15

            img.show()

        current_test_image += 1
        msg = "На изображении число " + str(np.argmax(layer_2))
        mb.showinfo("Информация", msg)


    