import cv2


class MockerVideoCapture:
    def __init__(self) -> None:
        self.video = None
        self.current = ""

    def set(self, path: str) -> None:
        if self.video is not None:
            self.video.release()

        self.video = cv2.VideoCapture(path)

        if not self.video.isOpened():
            raise ValueError("Unable to open video source", path)

        self.current = path

    def reset(self):
        self.video = None

    def get_frame(self) -> tuple[bool, cv2.Mat] | tuple[bool, None]:
        if self.video is not None and self.video.isOpened():
            ret, frame = self.video.read()
            if ret:
                return ret, frame
            else:
                return ret, None
        else:
            return False, None

    def __del__(self):
        if self.video is not None and self.video.isOpened():
            self.video.release()
