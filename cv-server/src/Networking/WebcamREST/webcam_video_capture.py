import cv2


class WebcamVideoCapture:
    def __init__(self) -> None:
        self.video = None
        self.current = ""

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

    def set(self, device_index: int = 0) -> None:
        """Initialize the webcam using the given device index (default is 0)."""
        self.reset()  # Release the previous one
        self.video = cv2.VideoCapture(device_index)
        if not self.video.isOpened():
            raise ValueError(f"Unable to open webcam with index {device_index}")
        self.current = f"Webcam {device_index}"

    def reset(self):
        """
        Release the video capture object.
        """
        if self.video is not None:
            self.video.release()
        self.video = None
        self.current = ""

    def get_frame(self) -> tuple[bool, cv2.Mat] | tuple[bool, None]:
        """
        Capture and return the current frame from the webcam.
        """
        if self.video is not None and self.video.isOpened():
            ret, frame = self.video.read()
            if ret:
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return ret, None
        else:
            return False, None

    def __del__(self):
        if self.video is not None and self.video.isOpened():
            self.video.release()
