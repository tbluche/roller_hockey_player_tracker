from dataclasses import dataclass
import struct
from math import sqrt
import matplotlib.patches


@dataclass
class Detection:
    startX: int
    startY: int
    endX: int
    endY: int
    confidence: float

    def box(self):
        return (startX, startY, endX, endY)

    def clone(self):
        return Detection(
            self.startX, self.startY, self.endX, self.endY, self.confidence
        )

    def union(self, other):
        startX = min(self.startX, other.startX)
        startY = min(self.startY, other.startY)
        endX = max(self.endX, other.endX)
        endY = max(self.endY, other.endY)
        return Detection(startX, startY, endX, endY, 0.0)

    def intersection(self, other):
        startX = max(self.startX, other.startX)
        startY = max(self.startY, other.startY)
        endX = min(self.endX, other.endX)
        endY = min(self.endY, other.endY)
        if endX <= startX or endY <= startY:
            return None
        return Detection(startX, startY, endX, endY, 0.0)

    def iou(self, other):
        intersect = self.intersection(other)
        if intersect is None:
            return 0.0
        union = self.union(other)
        return intersect.area() / union.area()

    def width(self):
        return self.endX - self.startX

    def height(self):
        return self.endY - self.startY

    def area(self):
        return self.width() * self.height()

    def middle(d):
        return (d.startX + d.endX) / 2, (d.startY + d.endY) / 2

    def distance(self, other):
        sx, sy = self.middle()
        ox, oy = other.middle()
        return sqrt((sx - ox) ** 2 + (sy - oy) ** 2)

    def crop(self, img_in):
        return img_in[self.startY : self.endY, self.startX : self.endX]

    def mpl_rect(self, **kwargs):
        return matplotlib.patches.Rectangle(
            (self.startX, self.startY), self.width(), self.height(), **kwargs
        )

    def write(self, f):
        f.write(struct.pack("i", self.startX))
        f.write(struct.pack("i", self.startY))
        f.write(struct.pack("i", self.endX))
        f.write(struct.pack("i", self.endY))
        f.write(struct.pack("f", self.confidence))

    @classmethod
    def load(cls, f):
        startX, startY, endX, endY = struct.unpack("iiii", f.read(16))
        confidence = struct.unpack("f", f.read(4))[0]
        return cls(startX, startY, endX, endY, confidence)


def save_detections(detections, filename):
    with open(filename, "wb") as f:
        f.write(struct.pack("i", len(detections)))
        for detection in detections:
            detection.write(f)


def load_detections(filename):
    with open(filename, "rb") as f:
        num = struct.unpack("i", f.read(4))[0]
        detections = [Detection.load(f) for _ in range(num)]
    return detections
