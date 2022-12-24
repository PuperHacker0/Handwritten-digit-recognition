#Importing necessary files for graphics interface and model

from PIL import Image, ImageDraw
import tkinter as tk
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
import os
import cv2
import numpy as np

class App():
    def __init__(self, root):
        self.root = root
        self.root.geometry('600x800')
        self.root.title("Digit recognition")
        self.create_label()
        self.create_canvas()
        self.create_buttons()
        self.root.resizable(width = False, height = False)
        self.image1 = Image.new("RGB", (600, 600), (255,255,255))
        self.carbonCopy = ImageDraw.Draw(self.image1)
        self.train_test_model()

    def ok_button_pushed(self): #Called when ok pressed
        image_file_name = self.extract_image()
        img_2 = self.resizing(image_file_name)
        prediction_number = self.read_predict_image(img_2)
        self.update_prediction(prediction_number)
        
    def extract_image(self):
        # PIL image can be saved as .png .jpg .gif or .bmp file (among others)
        filename = "img.jpg"
        self.image1.save(filename)
        return "img.jpg"
    
    def reset_button_pushed(self): #Called when reset pushed
        self.canvas.delete('all')
        self.image1 = Image.new("RGB", (600, 600), (255, 255, 255))
        self.carbonCopy = ImageDraw.Draw(self.image1)
        #self.canvas.create_image()

    def create_label(self):
        self.prediction_label = tk.Label(root, text= "Prediction number :" , font="Arial 24", bg="green",height = 3)
        self.prediction_label.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

    def update_prediction(self, prediction_number):
        self.prediction_label.config(text = f"Prediction number: " + str(prediction_number))
    def create_buttons(self):
        self.ok_button = tk.Button(root, text='OK', font='Arial 24', \
                              bg='Yellow', width=15, padx=5 , pady=5,command = self.ok_button_pushed)
        self.reset_button = tk.Button(root, text='Reset', font='Arial 24', \
                                 bg='Yellow', width=15,padx=5 , pady=5, command = self.reset_button_pushed)
        self.ok_button.grid(row=1, column=0, sticky="nsew")
        self.reset_button.grid(row=1, column=1, sticky="nsew")

    def create_canvas(self):
        self.canvas = tk.Canvas(self.root, width = 600, height = 600)
        self.canvas.grid(row = 0, column = 0, columnspan = 2)
        self.canvas.bind('<Button-1>', self.start_draw)
        self.canvas.bind('<B1-Motion>', self.draw_line)
        
    def start_draw(self, event):
        self.lastx, self.lasty = event.x, event.y
        
    def draw_line(self, event):
        self.canvas.create_line((self.lastx, self.lasty, event.x, event.y), width = 20)

        black = (0, 0, 0)
        self.carbonCopy.line([self.lastx, self.lasty, event.x, event.y], black, width = 20)
        self.lastx, self.lasty = event.x, event.y

    #define baseline model
    def baseline_model(self):
        # create model
        model1 = Sequential()
        model1.add(Dense(self.num_pixels, input_shape=(self.num_pixels,), kernel_initializer='normal', activation='relu'))
        model1.add(Dense(self.num_classes, kernel_initializer='normal', activation='softmax'))
        # Compile model
        model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model1

    def train_test_model(self):
        # load data
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        # flatten 28*28 images to a 784 vector for each image
        self.num_pixels = X_train.shape[1] * X_train.shape[2]
        X_train = X_train.reshape((X_train.shape[0], self.num_pixels)).astype('float32')
        X_test = X_test.reshape((X_test.shape[0], self.num_pixels)).astype('float32')
        # normalize inputs from 0-255 to 0-1
        X_train = X_train / 255
        X_test = X_test / 255
        # one hot encode outputs
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        self.num_classes = y_test.shape[1]
        # build the model
        self.model = self.baseline_model()
        # Fit the model
        self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=200, verbose=2)
        # Final evaluation of the model
        scores = self.model.evaluate(X_test, y_test, verbose=0)
        print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))

    def read_predict_image(self, img_2):
        if not os.path.isfile(img_2):
            print('Error')
            return

        img = cv2.imread(img_2)[:,:,0]
        img = np.invert(np.array([img]))
        img = np.reshape(img, (1, 28 * 28))
        
        prediction = self.model.predict(img)
        print(f"This is probably a {np.argmax(prediction)}")
        return np.argmax(prediction)

    def resizing(self,image_file_name):
        image = Image.open(image_file_name)
        image_file_name_2 = "image_file_name_2.jpeg"
        
        new_image = image.resize((28 , 28))
        new_image.save(image_file_name_2)
        return image_file_name_2



root = tk.Tk()
App(root)
root.mainloop()
