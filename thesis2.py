import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import matplotlib.pyplot as plt

class YoloApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YOLO Mulberry Counter")
        self.geometry("600x400")

        self.current_page = None
        self.pages = {}

        self.show_page("Home")

    def show_page(self, page_name, file_path=None):  # Accept an optional file_path argument
        if self.current_page is not None:
            self.current_page.pack_forget()

        if page_name == "Home":
            self.current_page = HomePage(self, self.show_page)
        elif page_name == "Webcam":
            self.current_page = WebcamPage(self, self.show_page)
        elif page_name == "Upload":
            self.current_page = UploadPage(self, self.show_page)
        elif page_name == "Result":
            self.current_page = ResultPage(self, self.show_page, file_path)  # Pass file_path

        self.current_page.pack(expand=True, fill="both")

class HomePage(tk.Frame):
    def __init__(self, master, show_page):
        super().__init__(master)
        self.master = master
        self.show_page = show_page

        label = tk.Label(self, text="Choose an option:")
        label.pack(pady=10)

        webcam_btn = tk.Button(self, text="Use Webcam", command=self.goto_webcam)
        webcam_btn.pack(pady=5)

        upload_btn = tk.Button(self, text="Upload Image", command=self.goto_upload)
        upload_btn.pack(pady=5)

    def goto_webcam(self):
        self.show_page("Webcam")

    def goto_upload(self):
        self.show_page("Upload")

class WebcamPage(tk.Frame):
    def __init__(self, master, show_page):
        super().__init__(master)
        self.master = master
        self.show_page = show_page

        self.video = cv2.VideoCapture(0)
        self.canvas = tk.Canvas(self, width=self.video.get(cv2.CAP_PROP_FRAME_WIDTH),
                                height=self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        count_btn = tk.Button(self, text="Count Mulberry", command=self.count_mulberry)
        count_btn.pack(pady=10)

        # Start the video feed
        self.update()

    def update(self):
        ret, frame = self.video.read()
        if ret:
            # Convert the frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert the frame to ImageTk format
            img = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            # Update the canvas with the new frame
            self.canvas.img = img
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
        # Schedule the update method to run after 10 milliseconds
        self.canvas.after(10, self.update)

    def count_mulberry(self):
        # Save the picture
        ret, frame = self.video.read()
        if ret:
            file_path = "webcam_capture.jpg"  # Choose a file path
            cv2.imwrite(file_path, frame)
            self.master.show_page("Result", file_path=file_path)

    def update(self):
        ret, frame = self.video.read()
        if ret:
            # Convert the frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert the frame to ImageTk format
            img = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            # Update the canvas with the new frame
            self.canvas.img = img
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
        # Schedule the update method to run after 10 milliseconds
        self.canvas.after(10, self.update)

class UploadPage(tk.Frame):
    def __init__(self, master, show_page):
        super().__init__(master)
        self.master = master
        self.show_page = show_page

        upload_btn = tk.Button(self, text="Upload Image", command=self.upload_image)
        upload_btn.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.master.show_page("Result", file_path)  # Pass file_path to ResultPage

class ResultPage(tk.Frame):
    def __init__(self, master, show_page, file_path):  # Receive file_path as an argument
        super().__init__(master)
        self.master = master
        self.show_page = show_page
        self.file_path = file_path  # Set file_path

        # Replace with your YOLO model setup
        self.model = YOLO(r'runs\detect\train3\weights\best.pt')
        # Replace with your YOLO results
        self.run_yolo()

        pie_chart_btn = tk.Button(self, text="Show Pie Chart", command=self.show_pie_chart)
        pie_chart_btn.pack(pady=10)

        home_btn = tk.Button(self, text="Home", command=self.goto_home)
        home_btn.pack(pady=10)

    def run_yolo(self):
        results = self.model.predict(self.file_path, imgsz=320, conf=0.3, save=True)
        self.yolo_results = results[0]

    def show_pie_chart(self):
        # Replace with your YOLO results processing
        labels = ['Flowering', 'Overripe', 'Ripe', 'Unripe']
        counts = [0, 0, 0, 0]

        for box in self.yolo_results.boxes:
            class_index = int(box.cls.item())  # Convert to integer
            if 0 <= class_index < len(labels):
                counts[class_index] += 1


        non_zero_labels = [label for label, count in zip(labels, counts) if count > 0]
        non_zero_counts = [count for count in counts if count > 0]

        plt.pie(non_zero_counts, labels=non_zero_labels, autopct=lambda p: '{:.0f}'.format(p * sum(counts) / 100))
        plt.show()



    def goto_home(self):
        self.master.show_page("Home")

if __name__ == "__main__":
    app = YoloApp()
    app.mainloop()
