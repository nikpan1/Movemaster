import uvicorn
import httpx
import os
import signal
from fastapi import FastAPI
from pydantic import BaseModel
from src.Base64.Base64Conversions import *
import cv2
import tkinter as tk
import PIL.Image, PIL.ImageTk
import threading
from src.MockerWebcamREST.catbase64 import cat_base64
from tkinter import filedialog
from tkinter import font as tkFont


class MockerWB(tk.Tk):
    def __init__(self, *args, **kwargs) -> None:
        tk.Tk.__init__(self, *args, **kwargs)
        self.video_capture = VideoCapture()
        self.playback_resolution = (640, 480)
        self.gui_setup()
        self.delay = 10
        self.playing = False
        self.webcam = False
        self.current_frame = None
        self.ret = False
        self.server = MockerCaptureCameraServer(self)
        self.server_thread = threading.Thread(target=self.server.start_server, daemon=True)
        self.server_thread.start()
        self.update()
        

    def gui_setup(self):
        self.wm_title("Webcam Mocker")
        self.configure(background="#333333")

        self.top_frame = tk.Frame(self, width=200, height=200, bg='grey')
        self.top_frame.grid(row=0, column=0, padx=5, pady=5)
        self.bottom_frame = tk.Frame(self, width=200, height=200, bg='grey')
        self.bottom_frame.grid(row=1, column=0, padx=5, pady=5)

        helv36 = tkFont.Font(family='Helvetica', size=16, weight=tkFont.BOLD)
        self.toggle_webcam_btn = tk.Button(
            self.bottom_frame,
            text='WEBCAM',
            command=self.toggle_webcam,
            font=helv36,
            background='#7E7E7E'
        )
        self.open_file_btn = tk.Button(
            self.bottom_frame,
            text='OPEN',
            command=self.select_file,
            font=helv36
        )
        self.play_btn = tk.Button(
            self.bottom_frame,
            text='PLAY',
            command=self.play_video,
            font=helv36
        )
        self.stop_btn = tk.Button(
            self.bottom_frame,
            text='STOP',
            command=self.stop_video,
            font=helv36
        )
        self.replay_btn = tk.Button(
            self.bottom_frame,
            text='REPLAY',
            command=self.replay_video,
            font=helv36
        )
        self.video_canvas = tk.Canvas(
            self.top_frame,
            width=self.playback_resolution[0],
            height=self.playback_resolution[1],
            background='black'
        )
        self.video_canvas.grid(row=0, column=0)
        self.toggle_webcam_btn.grid(row=0, column=0, padx=5, pady=5, ipadx=10)
        self.open_file_btn.grid(row=0, column=1, padx=5, pady=5, ipadx=10)
        self.replay_btn.grid(row=0, column=2, padx=5, pady=5, ipadx=10)
        self.play_btn.grid(row=0, column=3, padx=5, pady=5, ipadx=10)
        self.stop_btn.grid(row=0, column=4, padx=5, pady=5, ipadx=10)


    def update(self):
        if self.video_capture.video != None and self.playing:
            self.ret, frame = self.video_capture.get_frame()
            if self.ret:
                self.current_frame = frame
                image = PIL.Image.fromarray(frame)
                image = image.resize(self.playback_resolution, PIL.Image.Resampling.LANCZOS)
                self.photo = PIL.ImageTk.PhotoImage(image=image)
                self.video_canvas.create_image(0,0, image=self.photo, anchor=tk.NW)
            else:
                self.current_frame = None
                self.playing = False
        self.after(self.delay, self.update)

    def toggle_webcam(self):
        if self.webcam:
            self.toggle_webcam_btn.config(background="#7E7E7E")
            self.replay_btn["state"] = "normal"
            self.open_file_btn["state"] = "normal"
            self.play_btn["state"] = "normal"
            self.stop_btn["state"] = "normal"
            self.video_capture.reset()
            self.webcam = False
        else:
            self.toggle_webcam_btn.config(background="red")
            self.replay_btn["state"] = "disabled"
            self.open_file_btn["state"] = "disabled"
            self.play_btn["state"] = "disabled"
            self.stop_btn["state"] = "disabled"
            self.video_capture.set(0)
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
            initialdir=os.getcwd(),
            filetypes=[('MP4 files', '*.mp4')]
        )
        if filename != None:
            self.video_capture.set(filename.name)
            self.playing = False
            filename.close()

class VideoCapture():
    def __init__(self) -> None:
        self.video = None
        self.current = ""
    
    def set(self, path: str) -> None:
        if self.video != None:
            self.video.release()
        self.video = cv2.VideoCapture(path)
        if not self.video.isOpened():
            raise ValueError("Unable to open video source", path)
        self.current = path

    def reset(self):
        self.video = None

    def get_frame(self) -> tuple[bool, cv2.Mat] | tuple[bool, None]:
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

class Frame(BaseModel):
    image_base64: str

class MockerCaptureCameraServer:
    def __init__(self, mocker) -> None:
        self.UNITY_SERVER_URL = "http://localhost:7000"
        self.UNITY_SHUTDOWN_ENDPOINT = "/shutdown/"
        self.IS_UNITY_RUNNING = False
        self.app = FastAPI()
        self.mocker = mocker


    def start_server(self) -> None:
        self.setup_calls()
        uvicorn.run(self.app, host="127.0.0.1", port=8001)

    def shutdown_server(self) -> None:
        os.kill(os.getpid(), signal.SIGTERM)

    def setup_calls(self) -> None:
        @self.app.get("/health")
        async def health_check():
            self.IS_UNITY_RUNNING = True
            return {
                "status": "OK"
            }

        @self.app.post("/capture_and_send")
        async def capture_and_send():
            frame = mocker.current_frame
            if mocker.ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                image_base64 = image_to_base64(frame).decode('utf-8')
                return image_base64
            else:
                return cat_base64

        @self.app.post("/shutdown")
        async def shutdown():
            self.IS_UNITY_RUNNING = False
            self.shutdown_server()
            return {"message": "Capture Camera Server shutting down"}

        @self.app.on_event("shutdown")
        async def shutdown_event():
            if not self.IS_UNITY_RUNNING:
                async with httpx.AsyncClient() as client:
                    try:
                        await client.post(
                            self.UNITY_SERVER_URL + self.UNITY_SHUTDOWN_ENDPOINT,
                            json={"message": "Camera capture server is shutting down"}
                        )
                    except Exception as error:
                        print(f"Failed to send shutdown signal: {error}")

if __name__ == "__main__":
    mocker = MockerWB()
    mocker.mainloop()