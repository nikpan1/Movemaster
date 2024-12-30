import os
import threading
import tkinter as tk
from tkinter import filedialog
from tkinter import font as tk_font

import PIL.Image
import PIL.ImageTk
import torch

from ComputerVision.PoseEstimation.ExerciseClassification import Args, ExerciseRecognition
from Networking.MockerWebcamREST.mocker_video_capture import MockerVideoCapture
from Networking.MockerWebcamREST.webcam_mocker_server import MockerCaptureCameraServer


class WebcamMockerApplication(tk.Tk):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        args = Args()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.exercise_recognition = ExerciseRecognition(model_path=r"ComputerVision\PoseEstimation\model.pth",
                                                        repetitiveness=1, device=device, args=args)
        self.latest_predicted_class = "non_activity"
        self.latest_predicted_confidence = 0

        self.video_capture = MockerVideoCapture()
        self.playback_resolution = (640, 480)
        self.gui_setup()

        self.delay = 10
        self.playing = False
        self.webcam = False
        self.ret = False
        self.current_frame = MockerCaptureCameraServer.get_blank_image()
        self.photo = None

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

        helv36 = tk_font.Font(family='Helvetica', size=16, weight=tk_font.BOLD)
        self.open_file_btn = tk.Button(self.bottom_frame, text='OPEN', command=self.select_file, font=helv36)
        self.play_btn = tk.Button(self.bottom_frame, text='PLAY', command=self.play_video, font=helv36)
        self.stop_btn = tk.Button(self.bottom_frame, text='STOP', command=self.stop_video, font=helv36)
        self.replay_btn = tk.Button(self.bottom_frame, text='REPLAY', command=self.replay_video, font=helv36)
        self.video_canvas = tk.Canvas(self.top_frame, width=self.playback_resolution[0],
                                      height=self.playback_resolution[1], background='black')

        self.video_canvas.grid(row=0, column=0)
        self.open_file_btn.grid(row=0, column=1, padx=5, pady=5, ipadx=10)
        self.replay_btn.grid(row=0, column=2, padx=5, pady=5, ipadx=10)
        self.play_btn.grid(row=0, column=3, padx=5, pady=5, ipadx=10)
        self.stop_btn.grid(row=0, column=4, padx=5, pady=5, ipadx=10)

    def update(self):
        if self.video_capture.video is not None and self.playing:
            self.ret, frame = self.video_capture.get_frame()
            if self.ret:
                self.current_frame = frame
                self.latest_predicted_class, self.latest_predicted_confidence =\
                    self.exercise_recognition.recognize(frame)
                image = PIL.Image.fromarray(frame)
                image = image.resize(self.playback_resolution, PIL.Image.Resampling.LANCZOS)
                self.photo = PIL.ImageTk.PhotoImage(image=image)
                self.video_canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

            else:
                self.playing = False

        self.after(self.delay, self.update)

    def toggle_webcam(self):
        if self.webcam:
            self.replay_btn["state"] = "normal"
            self.open_file_btn["state"] = "normal"
            self.play_btn["state"] = "normal"
            self.stop_btn["state"] = "normal"
            self.video_capture.reset()
            self.webcam = False
        else:
            self.replay_btn["state"] = "disabled"
            self.open_file_btn["state"] = "disabled"
            self.play_btn["state"] = "disabled"
            self.stop_btn["state"] = "disabled"
            self.video_capture.set(0)
            self.playing = True
            self.webcam = True

    def play_video(self):
        if self.video_capture.video is not None:
            self.playing = True

    def stop_video(self):
        if self.video_capture.video is not None:
            self.playing = False

    def replay_video(self):
        if self.video_capture.video is not None:
            self.playing = True
            self.video_capture.set(self.video_capture.current)

    def select_file(self):
        filename = filedialog.askopenfile(title='Open a video file', initialdir=os.getcwd(),
                                          filetypes=[('MP4 files', '*.mp4')])
        if filename is not None:
            self.video_capture.set(filename.name)
            self.playing = False
            filename.close()

    def start_application(self):
        self.mainloop()
