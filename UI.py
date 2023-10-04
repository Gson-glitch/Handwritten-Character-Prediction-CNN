# Making Imports
from tkinter import Tk, Frame, Button, Label, Canvas, CENTER
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from string import ascii_uppercase
from tensorflow.keras.models import load_model


# Building the Graphical User Interface
root = Tk()
root.configure(background='#7FCDCD')
root.resizable(False, False)
img = None
page_title = "Handwritten Character Prediction using CNN"
img_path = None
category = "Please Upload an Image"
char_dict = dict(zip([i for i in range(26)], list(ascii_uppercase)))

def preprocess_img(img_path, threshold=120):
    # Load the image
    img = cv2.imread(img_path)
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize the image to 28x28 pixels
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    # Invert the colors
    inverted = cv2.bitwise_not(resized)
    # Removing Noise
    inverted[inverted <= threshold] = 0
    # Reshaping the array to fit the model architecture
    img_arr = inverted.reshape(1, 28, 28, 1)
    # Returning the image array
    return img_arr

def upload_img():
    global img, img_path, category
    image_filetypes = [
        ("JPG files", "*.jpg"),
        ("JPEG files", "*.jpeg"),
        ("PNG files", "*.png"),
        ("GIF files", "*.gif"),
        ("BMP files", "*.bmp")
    ]
    img_path = askopenfilename(
        title="Upload an Image File",
        filetypes=image_filetypes
    )
    if not img_path:
        return
    img_open = Image.open(img_path)

    # Calculate the aspect ratio of the image
    aspect_ratio = img_open.width / img_open.height
    # Set the maximum width and height of the canvas
    max_width = 400
    max_height = 200
    # Calculate the new width and height of the image
    if img_open.width > max_width or img_open.height > max_height:
        if aspect_ratio > 1:
            new_width = max_width
            new_height = max_width / aspect_ratio
        else:
            new_width = max_height * aspect_ratio
            new_height = max_height
    else:
        new_width = img_open.width
        new_height = img_open.height
    # Resize the image
    img_open = img_open.resize((int(new_width), int(new_height)), Image.LANCZOS)   
    img = ImageTk.PhotoImage(img_open)
    
    canvas = Canvas(master=image_frame, width=400, height=200)
    canvas.grid(row=0, column=1)
    canvas.create_image(max_width / 2, max_height / 2, anchor=CENTER, image=img)
    
    image_array = preprocess_img(img_path)
    caps_net = load_model(os.path.join(os.getcwd(), 'CNN', 'CNN.h5'))
    
    predicted_char = char_dict[caps_net.predict(image_array).argmax()]
    category = f"The predicted character is: {predicted_char}"
    prediction_label['text'] = category

# Creating Widgets
second_frame = Frame(master=root)
second_frame.grid(row=0,column=1)

first_frame = Frame(master=root)
first_frame.grid(row=0, column=0)

prediction_label = Label(master=second_frame, text=category, pady=3, padx=5, borderwidth=3)
prediction_label.grid(row=0, column=0, pady=10)

image_frame = Frame(master=second_frame, relief='groove', borderwidth=2, width=400, height=200, bg='#8EF0F7')
image_frame.grid(row=1, column=0)

picture_choose_btn = Button(master=first_frame, text='Click to Upload Picture.', width=20, bg='black', fg='#fff', command=upload_img)
picture_choose_btn.grid(row=1, column=0, pady=10)

# Customizing the Window
root.columnconfigure(0, minsize=100, weight=1)
root.rowconfigure(0, minsize=100, weight=1)
root.columnconfigure(1, minsize=500, weight=1)
root.rowconfigure(0, minsize=500, weight=1)
root.geometry("720x500")
root.title(page_title)

            
root.mainloop()

