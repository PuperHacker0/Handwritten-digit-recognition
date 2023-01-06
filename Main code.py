# Importing necessary files
# For graphics interface
from PIL import Image, ImageDraw
import tkinter as tk

# For model training
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# For image file input/output
import os
import cv2
import numpy as np


class Model():
    def __init__(self):
        # Setup model with basic parameters and compile it
        self.model = Sequential()
        self.num_pixels = 28 * 28
        self.num_classes = 10

        self.model.add(
            Dense(self.num_pixels, input_shape=(self.num_pixels,), kernel_initializer='normal', activation='relu'))
        self.model.add(Dense(self.num_classes, kernel_initializer='normal', activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def load_data(self):
        # Load data from dataset file
        (self.train_image_data, self.train_image_labels), (
        self.test_image_data, self.test_image_labels) = mnist.load_data()

        # Flatten 28*28 image data to 784 vector
        self.train_image_data = self.train_image_data.reshape((self.train_image_data.shape[0], self.num_pixels)).astype(
            'float32')
        self.test_image_data = self.test_image_data.reshape((self.test_image_data.shape[0], self.num_pixels)).astype(
            'float32')

        # Normalize input colors from range 0-255 to range 0-1
        self.train_image_data = self.train_image_data / 255
        self.test_image_data = self.test_image_data / 255

        # One hot encode outputs (results in range 0-9)
        self.train_image_labels = to_categorical(self.train_image_labels)
        self.test_image_labels = to_categorical(self.test_image_labels)

    def train(self):
        # Trains the model from the given dataset
        self.load_data()

        # Fit the data to the model and evaluate
        self.model.fit(self.train_image_data, self.train_image_labels,
                       validation_data=(self.test_image_data, self.test_image_labels), epochs=20, batch_size=200,
                       verbose=2)
        scores = self.model.evaluate(self.test_image_data, self.test_image_labels, verbose=0)
        print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))

    def predict(self, image_filename):
        # Read image from file and reshape it with corresponding dimensions to model
        query_image_data = cv2.imread(image_filename)[:, :, 0]
        query_image_data = np.invert(np.array([query_image_data]))
        query_image_data = np.reshape(query_image_data, (1, self.num_pixels))

        # Make prediction
        prediction = self.model.predict(query_image_data)
        return np.argmax(prediction)


class Application():
    def __init__(self, window):
        # Create window and its dimensions and title
        self.window = window
        self.window.geometry('600x800')
        self.window.resizable(width=False, height=False)
        self.window.title("Digit recognition")

        # Create the window's widgets
        self.create_label()
        self.create_canvas()
        self.create_buttons()

        # Create and train model
        self.model = Model()
        self.model.train()

    def create_label(self):
        # Create prediction label
        self.arial_font = "Arial 24"
        self.prediction_label = tk.Label(window, text="Prediction number :", \
                                         font=self.arial_font, height=3, bg="#c28dff", fg="purple")
        self.prediction_label.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

    def create_canvas(self):
        # Create canvas widget
        self.can_width = self.can_height = 600
        self.canvas = tk.Canvas(self.window, width=self.can_width, height=self.can_height)
        self.canvas.grid(row=0, column=0, columnspan=2)

        # Bind left mouse button actions for drawing on canvas
        self.canvas.bind('<Button-1>', self.start_line)
        self.canvas.bind('<B1-Motion>', self.draw_line)

        # Create canvas overlay
        self.whiteColorRGB = (255, 255, 255)
        self.overlayImage = Image.new("RGB", (self.can_width, self.can_height), self.whiteColorRGB)
        self.canvasOverlay = ImageDraw.Draw(self.overlayImage)

    def start_line(self, event):
        # Record coordinates of the first point drawn
        self.lastx, self.lasty = event.x, event.y

    def draw_line(self, event):
        # Draw line on canvas and overlay
        pen_thickness = 20
        black = (0, 0, 0)

        self.canvas.create_line((self.lastx, self.lasty, event.x, event.y), width=pen_thickness)
        self.canvasOverlay.line([self.lastx, self.lasty, event.x, event.y], black, width=20)

        # Update last drawn point with current point
        self.lastx, self.lasty = event.x, event.y

    def create_buttons(self):
        # Create ok button
        self.ok_button = tk.Button(window, text='OK', font=self.arial_font, bg='#9E4E96', \
                                   width=15, padx=5, pady=5, command=self.ok_button_pushed)
        self.ok_button.grid(row=1, column=0)

        # Create reset button
        self.reset_button = tk.Button(window, text='Reset', font=self.arial_font, bg='#9E4E96', \
                                      width=15, padx=5, pady=5, command=self.reset_button_pushed)
        self.reset_button.grid(row=1, column=1)

    def ok_button_pushed(self):
        # Save and resize canvas overlay image
        saved_image_filename = self.extract_can_overlay_image()
        resized_image_filename = self.resizing(saved_image_filename)

        # Make and update prediction for that image
        prediction = self.model.predict(resized_image_filename)
        self.prediction_label.config(text="Prediction: " + str(prediction))

    def resizing(self, saved_image_filename):
        # Open and resize image from canvas overlay to 28x28 pixels
        image = Image.open(saved_image_filename)
        resized_image = image.resize((28, 28))

        # Save resized image to file
        resized_image_filename = "resized_image.jpg"
        resized_image.save(resized_image_filename)
        return resized_image_filename

    def extract_can_overlay_image(self):
        # Saves overlay image to file and returns the filename
        filename = "drawn_number.jpg"
        self.overlayImage.save(filename)
        return filename

    def reset_button_pushed(self):
        # Erase canvas
        self.canvas.delete('all')

        # Erase canvas overlay
        self.overlayImage = Image.new("RGB", (self.can_width, self.can_height), self.whiteColorRGB)
        self.canvasOverlay = ImageDraw.Draw(self.overlayImage)


# Main
window = tk.Tk()
Application(window)
window.mainloop()
