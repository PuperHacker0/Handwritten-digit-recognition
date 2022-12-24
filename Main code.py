import tkinter as tk
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

class App():
    def __init__(self, root):
        self.root = root
        self.root.geometry('800x600')
        root.title("Digit recognition")
        self.create_canvas()
        self.create_buttons()

    def ok_button_pushed(self): #Called when ok pressed
        pass
    def reset_button_pushed(self): #Called when reset pushed
        self.canvas.delete('all')
        self.canvas.create_image()
    def create_buttons(self):
        self.ok_button = tk.Button(root, text='OK', font='Arial 24', \
                              bg='Yellow', width=15, command = self.ok_button_pushed)
        self.reset_button = tk.Button(root, text='Reset', font='Arial 24', \
                                 bg='Yellow', width=15, command = self.reset_button_pushed)

        self.ok_button.grid(row=1, column=0, sticky="nsew")
        self.reset_button.grid(row=1, column=1, sticky="nsew")

        # self.canvas_rows, self.canvas_columns = 2, 2
        # for i in range(self.canvas_rows):
        #     root.rowconfigure(root, i, weight = 1)
        # for j in range(self.canvas_columns):
        #     root.columnconfigure(root, j, weight = 1)

    def create_canvas(self):
        self.canvas = tk.Canvas(self.root, width = 800, height = 400)
        self.canvas.grid(row = 0, column = 0, columnspan = 2)
        self.canvas.bind('<Button-1>', self.start_draw)
        self.canvas.bind('<B1-Motion>', self.draw_line)
    def start_draw(self, event):
        self.lastx, self.lasty = event.x, event.y
    def draw_line(self, event):
        self.canvas.create_line((self.lastx, self.lasty, event.x, event.y), width = 20)
        self.lastx, self.lasty = event.x, event.y

    #define baseline model
    def baseline_model(self):
        # create model
        model = Sequential()
        model.add(Dense(num_pixels, input_shape=(num_pixels,), kernel_initializer='normal', activation='relu'))
        model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    def run_digit(self):
        # load data
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        # flatten 28*28 images to a 784 vector for each image
        num_pixels = X_train.shape[1] * X_train.shape[2]
        X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
        X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')
        # normalize inputs from 0-255 to 0-1
        X_train = X_train / 255
        X_test = X_test / 255
        # one hot encode outputs
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        num_classes = y_test.shape[1]

        # build the model
        model = baseline_model()
        # Fit the model
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
        # Final evaluation of the model
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))


root = tk.Tk()
App(root)
root.mainloop()
