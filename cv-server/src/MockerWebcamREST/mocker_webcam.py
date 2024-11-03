# import uvicorn
# import httpx
# import os
# import signal
# from fastapi import FastAPI, BackgroundTasks
# from pydantic import BaseModel
# from src.Base64.Base64Conversions import *
import cv2

class VideoPlayer:
    def __init__(self) -> None:
        self.cap = None
        self.name = None
        self.stop = False
    
    def set(self, path: str) -> None:
        if (self.cap != None):
            self.cap.release()
        print(path)
        self.cap = cv2.VideoCapture(path)
        self.name = path

    def play(self) -> None:
        while(not self.stop and self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret == True:
                cv2.imshow('Frame',frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
        self.cap.release()

video_player = VideoPlayer()
video_player.set('videos/BWSquat.mp4')
video_player.play()
cv2.destroyAllWindows()