import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

car_cascade_path = r'.\models\haarcascade_car.xml'
people_prototxt_path = r'.\models\MobileNetSSD_deploy.prototxt'
people_model_path = r'.\models\MobileNetSSD_deploy.caffemodel'
car_color_model_path = r'.\models\car_color_model.h5'

car_cascade = cv2.CascadeClassifier(car_cascade_path)
if car_cascade.empty():
    raise Exception("Failed to load car cascade classifier.")

people_net = cv2.dnn.readNetFromCaffe(people_prototxt_path, people_model_path)
car_color_model = load_model(car_color_model_path)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
           "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
           "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def detect_cars_and_people(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    car_count = len(cars)
    print(f'Detected {car_count} cars')

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 
                                 scalefactor=0.007843, size=(300, 300), mean=127.5)
    people_net.setInput(blob)
    detections = people_net.forward()

    people_count = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        idx = int(detections[0, 0, i, 1])
        if confidence > 0.4 and CLASSES[idx] == "person":
            people_count += 1
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    for (x, y, w_c, h_c) in cars:
        car_image = image[y:y + h_c, x:x + w_c]
        car_img_resized = cv2.resize(car_image, (64, 64))
        car_img_array = np.expand_dims(car_img_resized / 255.0, axis=0)

        predicted_color = car_color_model.predict(car_img_array)
        predicted_class = np.argmax(predicted_color, axis=1)

        color_rectangle = (0, 0, 255) if predicted_class[0] == 1 else (255, 0, 0)
        color_text = "Blue Car" if predicted_class[0] == 1 else "Other Car"

        cv2.rectangle(image, (x, y), (x + w_c, y + h_c), color_rectangle, 2)
        cv2.putText(image, color_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_rectangle, 2)

    return image, car_count, people_count

class CarColorDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Car Color and People Detection")

        self.label = tk.Label(root, text="Load an image of traffic")
        self.label.pack(pady=10)

        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack(pady=5)

        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return

        image = cv2.imread(file_path)
        if image is None:
            messagebox.showerror("Error", "Failed to load image.")
            return

        processed_image, car_count, people_count = detect_cars_and_people(image)

        img_height, img_width = processed_image.shape[:2]
        ratio = min(800 / img_width, 600 / img_height)
        new_size = (int(img_width * ratio), int(img_height * ratio))
        processed_image = cv2.resize(processed_image, new_size)

        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

        self.display_image(processed_image)
        messagebox.showinfo("Results", f"Cars Detected: {car_count}\nPeople Detected: {people_count}")

    def display_image(self, img):
        img = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(img)

        self.canvas.create_image(400, 300, anchor=tk.CENTER, image=img_tk)
        self.canvas.image = img_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = CarColorDetectionApp(root)
    root.mainloop()
