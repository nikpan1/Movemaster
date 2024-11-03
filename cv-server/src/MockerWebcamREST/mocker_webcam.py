# import uvicorn
# import httpx
import os
# import signal
# from fastapi import FastAPI, BackgroundTasks
# from pydantic import BaseModel
# from src.Base64.Base64Conversions import *
import cv2
import tkinter as tk
import PIL.Image, PIL.ImageTk
from tkinter import filedialog


class MockerUI(tk.Tk):
    def __init__(self, *args, **kwargs) -> None:
        tk.Tk.__init__(self, *args, **kwargs)
        self.video_capture = VideoCapture()
        self.gui_setup()
        self.delay = 10
        self.update()

    def gui_setup(self):
        self.wm_title("Webcam Mocker")
        self.open_file_btn = tk.Button(
            self,
            text = 'Open a video file',
            command = self.select_file
        )
        self.start_btn = tk.Button(
            self,
            text = 'START',
            command = None,
        )
        self.stop_btn = tk.Button(
            self,
            text = 'STOP',
            command = None,
        )
        self.replay_btn = tk.Button(
            self,
            text = 'REPLAY',
            command = self.replay_video,
        )
        self.video_canvas = tk.Canvas(
            self
        )
        self.video_canvas.pack()
        self.open_file_btn.pack()
        self.replay_btn.pack()


    def update(self):
        if self.video_capture.video != None:
            ret, frame = self.video_capture.get_frame()
            if ret:
                self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
                self.video_canvas.create_image(0,0, image = self.photo, anchor = tk.NW)
        self.after(self.delay, self.update)

    def replay_video(self):
        self.video_capture.set(self.video_capture.current)

    def select_file(self):
        filename = filedialog.askopenfile(
            title='Open a video file',
            initialdir = os.getcwd(),
            filetypes = [('MP4 files', '*.mp4')]
        )
        if filename != None:
            self.video_capture.set(filename.name)
            self.resize_video_canvas()
            filename.close()

    def resize_video_canvas(self):
        self.video_canvas.config(
            width = self.video_capture.width,
            height = self.video_capture.height
        )

class VideoCapture():
    def __init__(self) -> None:
        self.video = None
        self.current = ""
        self.width = 0
        self.height = 0
    
    def set(self, path: str) -> None:
        if self.video != None:
            self.video.release()
        self.video = cv2.VideoCapture(path)
        if not self.video.isOpened():
            raise ValueError("Unable to open video source", path)
        self.width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.current = path

    def get_frame(self):
        if self.video != None and self.video.isOpened():
            ret, frame = self.video.read()
            if ret == True:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (False, None)
        
    def __del__(self):
        if self.video != None and self.video.isOpened():
            self.video.release()

mockerUI = MockerUI()
mockerUI.mainloop()