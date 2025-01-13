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

    @staticmethod
    def list_available_cameras(max_tested=5):
        """
        Checks which camera indices are available in the range [0..max_tested-1].
        """
        available = []
        for idx in range(max_tested):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                available.append((idx, f"Webcam {idx}"))
                cap.release()
        return available

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
