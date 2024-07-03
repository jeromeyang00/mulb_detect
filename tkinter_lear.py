from tkinter import *
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
#calling the window

class YoloApp(Tk):
    def __init__(self):
        super().__init__()
        self.title("YOLO Mulberry Counter")
        self.geometry("800x400") #the witdh is 800 and the height is 400
        self.current_page = None
        self.pages = {}
        self.show_page("Camera")

    def show_page(self, page_name, file_path=None):  # Accept an optional file_path argument
        if self.current_page is not None:
            self.current_page.pack_forget()

        if page_name == "Camera":
            self.current_page = CameraPage(self, self.show_page)
        elif page_name == "Preview":
            self.current_page = PreviewPage(self, self.show_page, file_path)
        elif page_name == "Result":
            self.current_page = ResultPage(self, self.show_page, file_path)

        self.current_page.pack(expand=True, fill="both")

class CameraPage(Frame):
    def __init__(self, master, show_page):
        super().__init__(master)
        self.master = master
        self.show_page = show_page

        self.video = None  # Initialize video capture object
        self.canvas = Canvas(self, width=700, height=300, bg="black")
        self.canvas.pack()

        count_btn = Button(self, text="CAPTURE IMAGE", command=self.count_mulberry,  width=30, height=3, bg="red", fg="white", font=('Prestige Elite Std', 7, 'bold')) 
        #Button(self, text="RECAPTURE IMAGE", command=self.back, width=30, height=3, bg="red", fg="white", font=('Prestige Elite Std', 7, 'bold'))
        count_btn.pack(pady=10)

        # Start the video feed
        self.start_video_feed()

    def start_video_feed(self):
        if self.video is not None:
            self.video.release()  # Release the video capture object if it exists

        self.video = cv2.VideoCapture(0)  # Create a new video capture object
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

            # Calculate the center coordinates of the canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            image_width = img.width()
            image_height = img.height()
            x = (canvas_width - image_width) // 2
            y = (canvas_height - image_height) // 2

            self.canvas.create_image(x, y, anchor=NW, image=img)
        # Schedule the update method to run after 10 milliseconds

        self.canvas.after(10, self.update)

    def count_mulberry(self):
        # Save the picture
        ret, frame = self.video.read()
        if ret:
            file_path = "webcam_capture.jpg"  # Choose a file path
            cv2.imwrite(file_path, frame)
            self.master.show_page("Preview", file_path=file_path)

    def __del__(self):
        if self.video is not None:
            self.video.release() 
    

class PreviewPage(Frame):
    def __init__(self, master, show_page, file_path):
        super().__init__(master)
        self.master = master
        self.show_page = show_page

        self.file_path = file_path
        #self.create_widgets()
        # self.image_canvas.pack()
        #self.image_canvas = Canvas(self, bg="darkgray")
        self.image_canvas = Canvas(self, bg="darkgray",height=200, width=700)

        #resize the image
        

        self.image_canvas.pack()
        
        image_label = Label(self, text="IMAGE CAPTURED", font=('Prestige Elite Std', 7, 'bold'))
        image_label.pack(pady=10)

        count_btn = Button(self, command=self.count_mulberry,text="COUNT", width=30, height=3, bg="green", fg="lightgreen", font=('Prestige Elite Std', 7, 'bold'))
        count_btn.pack(pady=2)

        camera_btn = Button(self, text="RECAPTURE IMAGE", command=self.back, width=30, height=3, bg="red", fg="#FF7F7F", font=('Prestige Elite Std', 7, 'bold'))
        camera_btn.pack(pady=2)

        self.create_widgets()
    def create_widgets(self):

        print("FULL FILE PATH IS:", self.file_path)
        #extracts the file name of the image
        file_name = self.file_path.split("/")[-1].split(".")[0]
        print("FILENAME IS:", file_name)

        image_pil = Image.open(self.file_path)

        #resize the image
        image_pil.thumbnail((320, 240))
        image_tk = ImageTk.PhotoImage(image_pil)
        self.image_canvas.image = image_tk  # Store the image reference to prevent garbage collection

        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        print("CANVAS DIMENSION",canvas_width, canvas_height)
        image_width = image_tk.width()
        image_height = image_tk.height()
        x = (canvas_width - image_width) // 2
        y = (canvas_height - image_height) // 2
        self.image_canvas.create_image(200,0, anchor=NW,image=image_tk)

#_tkinter.TclError: bad anchor position "C": must be n, ne, e, se, s, sw, w, nw, or center

    def count_mulberry(self):
        self.master.show_page("Result", file_path=self.file_path)

    def back(self):
        self.master.show_page("Camera")
        self.master.destroy()
        app = YoloApp()
        app.mainloop()

class ResultPage(Frame):
    def __init__(self, master, show_page, file_path):  # Receive file_path as an argument
        super().__init__(master)
        self.master = master
        self.show_page = show_page
        self.file_path = file_path  # Set file_path

        # Replace with your YOLO model setup
        self.model = YOLO(r'runs\detect\train3\weights\best.pt')
        # Replace with your YOLO results
        self.run_yolo()

        self.create_widgets()

    def run_yolo(self):
        results = self.model.predict(self.file_path, imgsz=320, conf=0.3, save=True)
        self.yolo_results = results[0]

    def create_widgets(self):
        # Create the first column to display the list of classes detected with the number
        class_frame = Frame(self)
        class_frame.pack(side=LEFT, padx=10)

        class_label = Label(class_frame, text="YIELD QUANTIFICATION", font=('Prestige Elite Std', 15, 'bold'), bg="darkgray", fg="white")
        class_label.pack(pady=10)

        
        camera_btn = Button(self, text="RECAPTURE IMAGE", command=self.back, width=30, height=3, bg="red", fg="white", font=('Prestige Elite Std', 7, 'bold'))
        camera_btn.pack(pady=2)

        # for box in self.yolo_results.boxes:
            # class_index = int(box.cls.item())  # Convert to integer
            # class_name = self.model.names[class_index]
            # class_count = self.yolo_results.get_counts(class_index)

            # class_text = f"{class_name}: {class_count}"
            # class_text_label = tk.Label(class_frame, text=class_text)
            # class_text_label.pack()]

        labels = ['Flowering', 'Overripe', 'Ripe', 'Unripe']
        counts = [0, 0, 0, 0]
        image_results_path = f'{self.yolo_results.save_dir}'
        print("file path: ",image_results_path)

        for box in self.yolo_results.boxes:
            class_index = int(box.cls.item())  # Convert to integer
            if 0 <= class_index < len(labels):
                counts[class_index] += 1
                print(f"Class: {labels[class_index]}, Confidence: {box.conf.item()}")

        for i, label in enumerate(labels):
            class_text = f"{label}: {counts[i]}"
            class_text_label = Label(class_frame, text=class_text)
            class_text_label.pack()

        print("FULL FILE PATH IS:", self.file_path)
        #extracts the file name of the image
        file_name = self.file_path.split("/")[-1].split(".")[0]
        print("FILENAME IS:", file_name)

        # Create the second column to showcase the image detected
        image_frame = Frame(self)
        image_frame.pack(side=RIGHT, padx=10)

        image_label = Label(image_frame, text="IMAGE DETECTED", font=('Prestige Elite Std', 15, 'bold'), bg="darkgray", fg="white")
        image_label.pack(pady=10)
        
        # Get the absolute path of the image file
        image_pil = Image.open(image_results_path+f'\\{file_name}'+'.jpg')
        image_pil.thumbnail((320, 240))  # Resize the image to fit the frame
        image_tk = ImageTk.PhotoImage(image_pil)

        image_canvas = Canvas(image_frame, width=600, height=400)
        image_canvas.image = image_tk  # Store the image reference to prevent garbage collection
        
        canvas_width = image_canvas.winfo_width()
        canvas_height = image_canvas.winfo_height()
        print("CANVAS DIMENSION",canvas_width, canvas_height)
        image_width = image_tk.width()
        image_height = image_tk.height()
        x = (canvas_width - image_width) // 2
        y = (canvas_height - image_height) // 2


        image_canvas.create_image(100, 0, anchor=NW, image=image_tk)
        image_canvas.pack()


    def back(self):
        self.master.show_page("Camera")
        self.master.destroy()
        app = YoloApp()
        app.mainloop()

    # def capture_image(self):
    #     cap = cv2.VideoCapture(0)
    #     ret, frame = cap.read()
    #     if ret:
    #         cv2.imwrite("image.jpg", frame)
    #     cap.release()

# class PreviewPage(Frame):
#     def __init__(self, master, show_page, file_path):
#         super().__init__(master)
#         # Add your code for the PreviewPage here

# class ResultPage(Frame):
#     def __init__(self, master, show_page, file_path):
#         super().__init__(master)
#         # Add your code for the ResultPage here

if __name__ == "__main__":
    app = YoloApp()
    app.mainloop()


# first_frame = Frame(master=window, width=800, height=300) #relief is the border style of the frame sunken is the border style where the border is sunken into the frame
# #input the camera feed into the first_frame
# second_frame = Frame(master=window, width=800, height=100)

# first_frame.pack()
# second_frame.pack()



# def show_camera():
#     ret, frame = cap.read()
#     if ret:
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame = Image.fromarray(frame)
#         frame = ImageTk.PhotoImage(frame)
#         camera_label.imgtk = frame
#         camera_label.configure(image=frame)
#         camera_label.after(10, show_camera)

# camera_label = Label(master=first_frame, width=400, height=300, background="maroon")
# show_camera()
# if cap is None:
#     capture_image.state(["disabled"])
# camera_label.pack(padx=1,pady=1)

# capture_image = Button(
#     first_frame,text="CAPTURE IMAGE",
#     bg="green", 
#     fg="lightgreen",
#     font=('Prestige Elite Std', 15, 'bold'),
#     width=35,
#     height=3)
# capture_image.pack()

#change the color of the button into green


