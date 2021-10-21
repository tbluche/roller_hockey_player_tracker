from torchvision.models import detection
import numpy as np
import torch
import cv2
from .detection import Detection

DEVICE = torch.device("cpu")
NUM_CLASSES = 91
PERSON = 1
MODELS = {
    "frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
    "frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
    "retinanet": detection.retinanet_resnet50_fpn,
}


class PlayerDetector:
    """
    Detect players on the rink using a pre-trained Neural Network on the
    COCO classes.

    From an image file, the nnet is run and the detections corresponding to the
    `person` class are kept along with the confidence.
    """

    def __init__(self, model_key="frcnn-resnet"):
        self.model = MODELS[model_key](
            pretrained=True,
            progress=True,
            num_classes=NUM_CLASSES,
            pretrained_backbone=True,
        ).to(DEVICE)
        self.model.eval()

    def image_to_tensor(self, img_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        image = torch.FloatTensor(image)
        image = image.to(DEVICE)
        return image

    def detect_people(self, image_t):
        detections = self.model(image_t)[0]
        detected_people = []
        for i in range(0, len(detections["boxes"])):
            confidence = detections["scores"][i]
            idx = int(detections["labels"][i])
            if idx != PERSON:
                continue
            box = detections["boxes"][i].detach().cpu().numpy()
            (startX, startY, endX, endY) = box.astype("int")
            detected_people.append(Detection(startX, startY, endX, endY, confidence))
        return detected_people
