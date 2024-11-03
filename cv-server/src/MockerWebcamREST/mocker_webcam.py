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
from tkinter import font as tkFont


class MockerWB(tk.Tk):
    def __init__(self, *args, **kwargs) -> None:
        tk.Tk.__init__(self, *args, **kwargs)
        self.video_capture = VideoCapture()
        self.gui_setup()
        self.delay = 10
        self.playing = False
        self.webcam = False
        self.update()

    def gui_setup(self):
        self.wm_title("Webcam Mocker")
        self.configure(background="#333333")

        self.top_frame = tk.Frame(self, width=600, height=200, bg='grey')
        self.top_frame.grid(row=0, column=0, padx=5, pady=5)
        self.bottom_frame = tk.Frame(self, width=200, height=200, bg='grey')
        self.bottom_frame.grid(row=1, column=0, padx=5, pady=5)

        helv36 = tkFont.Font(family='Helvetica', size=16, weight=tkFont.BOLD)
        self.toggle_webcam_btn = tk.Button(
            self.bottom_frame,
            text = 'WEBCAM',
            command = self.toggle_webcam,
            font = helv36,
            background = '#7E7E7E'
        )
        self.open_file_btn = tk.Button(
            self.bottom_frame,
            text = 'OPEN',
            command = self.select_file,
            font = helv36
        )
        self.play_btn = tk.Button(
            self.bottom_frame,
            text = 'PLAY',
            command = self.play_video,
            font = helv36
        )
        self.stop_btn = tk.Button(
            self.bottom_frame,
            text = 'STOP',
            command = self.stop_video,
            font = helv36
        )
        self.replay_btn = tk.Button(
            self.bottom_frame,
            text = 'REPLAY',
            command = self.replay_video,
            font = helv36
        )
        self.video_canvas = tk.Canvas(
            self.top_frame,
        )
        self.video_canvas.grid(row=0, column=0)
        self.toggle_webcam_btn.grid(row=0, column=0, padx=5, pady=5, ipadx=10)
        self.open_file_btn.grid(row=0, column=1, padx=5, pady=5, ipadx=10)
        self.replay_btn.grid(row=0, column=2, padx=5, pady=5, ipadx=10)
        self.play_btn.grid(row=0, column=3, padx=5, pady=5, ipadx=10)
        self.stop_btn.grid(row=0, column=4, padx=5, pady=5, ipadx=10)


    def update(self):
        if self.video_capture.video != None and self.playing:
            ret, frame = self.video_capture.get_frame()
            if ret:
                self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
                self.video_canvas.create_image(0,0, image = self.photo, anchor = tk.NW)
            else:
                self.playing = False
        self.after(self.delay, self.update)

    def toggle_webcam(self):
        if self.webcam:
            self.toggle_webcam_btn.config(background = "#7E7E7E")
            self.replay_btn["state"] = "normal"
            self.open_file_btn["state"] = "normal"
            self.play_btn["state"] = "normal"
            self.stop_btn["state"] = "normal"
            self.video_capture.reset()
            self.webcam = False
        else:
            self.toggle_webcam_btn.config(background = "red")
            self.replay_btn["state"] = "disabled"
            self.open_file_btn["state"] = "disabled"
            self.play_btn["state"] = "disabled"
            self.stop_btn["state"] = "disabled"
            self.video_capture.set(0)
            self.resize_video_canvas()
            self.playing = True
            self.webcam = True

    def play_video(self):
        if self.video_capture.video != None:
            self.playing = True
    
    def stop_video(self):
        if self.video_capture.video != None:
            self.playing = False

    def replay_video(self):
        if self.video_capture.video != None:
            self.playing = True
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
            self.playing = False
            filename.close()

    def resize_video_canvas(self):
        self.video_canvas.config(
            width = self.video_capture.width,
            height = self.video_capture.height,
            background = "black"
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

    def reset(self):
        self.video = None

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

mocker_webcam = MockerWB()
mocker_webcam.mainloop()