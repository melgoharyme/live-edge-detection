import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Gohary Edge Detection")
        self.geometry("960x720")
        self.cap = cv2.VideoCapture(0)
        self.current_mode = 'o'
        self.sigma = 1.0

        # Configure grid layout
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Main video label
        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")
        self.video_label.configure(fg_color="black", corner_radius=15)

        # Filter name label
        self.filter_label = ctk.CTkLabel(self.video_label, text="Original", font=("Arial", 20), text_color="white", fg_color="#333", corner_radius=8)
        self.filter_label.place(relx=0.5, rely=0.05, anchor="n")

        # Controls frame - positioned in the middle of the bottom
        self.controls_frame = ctk.CTkFrame(self)
        self.controls_frame.grid(row=1, column=0, pady=10, sticky="")

        # Center the controls frame contents
        self.controls_frame.grid_columnconfigure(0, weight=1)
        self.controls_frame.grid_columnconfigure(1, weight=1)
        self.controls_frame.grid_columnconfigure(2, weight=1)

        # Filter menu (centered)
        self.filters = {
            "Original (Key: O)": 'o',
            "Sobel X (Key: X)": 'x',
            "Sobel Y (Key: Y)": 'y',
            "Magnitude (Key: M)": 'm',
            "Sobel + Thresh (Key: S)": 's',
            "LoG (Key: L)": 'l'
        }

        self.filter_menu = ctk.CTkOptionMenu(
            self.controls_frame, 
            values=list(self.filters.keys()), 
            command=self.on_filter_select
        )
        self.filter_menu.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

        # Exit button (to the right of the menu)
        self.exit_button = ctk.CTkButton(
            self.controls_frame, 
            text="Exit (Key: Q)", 
            command=self.quit,
            fg_color="red", 
            text_color="white", 
            corner_radius=8
        )
        self.exit_button.grid(row=0, column=2, padx=10, pady=5, sticky="e")

        # Key bindings
        self.bind("<Key>", self.handle_key_event)
        self.bind("<FocusIn>", lambda e: self.focus_set())
        self.focus_set()
        
        self.update_video()

    def handle_key_event(self, event):
        key = event.char.lower()
        
        if key == 'q':
            self.quit()
        elif key == '+':
            self.adjust_sigma(0.2)
        elif key == '-':
            self.adjust_sigma(-0.2)
        elif key in ['o', 'x', 'y', 'm', 's', 'l']:
            self.set_mode(key)
        
        self.focus_set()

    def on_filter_select(self, choice):
        self.current_mode = self.filters.get(choice, 'o')
        self.filter_label.configure(text=choice.split(" (")[0])  # Remove key hint from displayed text
        self.focus_set()

    def set_mode(self, mode):
        for name, val in self.filters.items():
            if val == mode:
                self.current_mode = val
                self.filter_label.configure(text=name.split(" (")[0])  # Remove key hint
                break
        self.focus_set()

    def adjust_sigma(self, delta):
        self.sigma = max(0.2, self.sigma + delta)
        self.focus_set()

    def update_video(self):
        ret, frame = self.cap.read()
        if not ret:
            self.after(10, self.update_video)
            return

        if self.current_mode != 'o':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (0, 0), self.sigma)
        else:
            frame = cv2.GaussianBlur(frame, (0, 0), self.sigma)
            blurred = frame

        if self.current_mode == 'x':
            result = cv2.convertScaleAbs(cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3))
        elif self.current_mode == 'y':
            result = cv2.convertScaleAbs(cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3))
        elif self.current_mode == 'm':
            gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            mag = np.sqrt(gx ** 2 + gy ** 2)
            result = cv2.convertScaleAbs(mag)
        elif self.current_mode == 's':
            gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            mag = np.sqrt(gx ** 2 + gy ** 2)
            result = cv2.threshold(cv2.convertScaleAbs(mag), 100, 255, cv2.THRESH_BINARY)[1]
        elif self.current_mode == 'l':
            log = cv2.Laplacian(blurred, cv2.CV_64F)
            result = cv2.convertScaleAbs(log)
        else:
            result = frame

        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        else:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(result)
        img = img.resize((self.video_label.winfo_width(), self.video_label.winfo_height()))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.configure(image=imgtk)
        self.video_label.image = imgtk

        self.after(20, self.update_video)

    def quit(self):
        self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()
