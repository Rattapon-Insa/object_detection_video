import torch
import numpy as np
import cv2
import pafy

class ObjectDetection:
    def __init__(self, url, out_file):
        self._URL = url
        self.model = self.load_model()
        self.classes = self.model.names
        self.out_file = out_file
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        print("\n\nDevice used: ", self.device)

    def get_video(self):

        play = pafy.new(self._URL).streams[-1]
        assert play is not None # check if pkay is not null
        return cv2.VideoCapture(play.url)

    def load_model(self):

        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained = True)
        return model
    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        result = self.model(frame)


