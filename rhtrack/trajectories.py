import struct
import matplotlib.pyplot as plt
from .detection import Detection


def interpolate(x0, y0, x1, y1, x):
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


def interpolate_detection(x0, d0, x1, d1, x):
    return Detection(
        startX=interpolate(x0, d0.startX, x1, d1.startX, x),
        startY=interpolate(x0, d0.startY, x1, d1.startY, x),
        endX=interpolate(x0, d0.endX, x1, d1.endX, x),
        endY=interpolate(x0, d0.endY, x1, d1.endY, x),
        confidence=interpolate(x0, d0.confidence, x1, d1.confidence, x),
    )


class Trajectory:
    def __init__(self):
        self.detections = []
        self.scores = []

    def __iter__(self):
        for item in self.detections:
            yield item

    def __len__(self):
        return len(self.detections)

    def __getitem__(self, idx):
        return self.detections[idx]

    def crop(self, start_frame, end_frame):
        new_trajectory = Trajectory()
        for (frame_idx, detection), score in zip(self, self.scores):
            if frame_idx < start_frame:
                continue
            if frame_idx > end_frame:
                break
            new_trajectory.add(frame_idx, detection, score=score)
        return new_trajectory

    def interpolate(self):
        duration = self.end() - self.start() + 1
        if len(self.detections) == duration:
            return self
        new_trajectory = Trajectory()
        prev = None
        prev_idx = 0
        prev_score = 0.0
        for (frame_idx, detection), score in zip(self, self.scores):
            if prev is None or frame_idx == prev_idx + 1:
                new_trajectory.add(frame_idx, detection, score=score)
            else:
                num_interpolation = frame_idx - prev_idx - 1
                for i in range(num_interpolation):
                    int_idx = prev_idx + i + 1
                    int_score = interpolate(
                        prev_idx, prev_score, frame_idx, score, int_idx
                    )
                    int_detection = interpolate_detection(
                        prev_idx, prev, frame_idx, detection, int_idx
                    )
                    new_trajectory.add(int_idx, int_detection, score=int_score)
                new_trajectory.add(frame_idx, detection, score=score)
            prev = detection
            prev_idx = frame_idx
            prev_score = score
        return new_trajectory

    def start(self):
        return self.detections[0][0]

    def end(self):
        return self.detections[-1][0]

    def latest(self):
        return self.detections[-1][1]

    def first(self):
        return self.detections[0][1]

    def score(self, normed=True):
        if not normed:
            return sum(self.scores)
        return self.score(normed=False) / len(self)

    def add(self, frame_idx, detection, score=0.0):
        self.detections.append((frame_idx, detection.clone()))
        self.scores.append(score)

    def clone(self):
        traj = Trajectory()
        for (frame_idx, detection), score in zip(self, self.scores):
            traj.add(frame_idx, detection, score=score)
        return traj

    def merge(self, other):
        assert other.start() >= self.end()
        traj = self.clone()
        for di, (frame_idx, detection) in enumerate(other):
            traj.add(frame_idx, detection, score=other.scores[di])
        return traj

    def add_copy(self, frame_idx, detection, score=0.0):
        traj = self.clone()
        traj.add(frame_idx, detection, score=score)
        return traj

    def plot(self, max_idx=None, color="r", label="Player"):
        detections = [
            detection
            for frame_idx, detection in self
            if max_idx is None or frame_idx <= max_idx
        ]
        last_fi = [
            frame_idx
            for frame_idx, detection in self
            if max_idx is None or frame_idx <= max_idx
        ][-1]
        if len(detections) < 1:
            return
        last = detections[-1]
        if max_idx is None or max_idx == last_fi:
            rect = last.mpl_rect(linewidth=2, edgecolor=color, facecolor="none")
            plt.gca().add_patch(rect)
            plt.text(
                last.startX,
                last.startY,
                f"{label}",
                color=color,
                fontsize=16,
                va="bottom",
            )
        xy = [d.middle() for d in detections]
        x, y = zip(*xy)
        plt.plot(x, y, "-", color=color, lw=3)

    def write(self, f):
        f.write(struct.pack("i", len(self.detections)))
        for (frame_idx, detection), score in zip(self.detections, self.scores):
            f.write(struct.pack("i", frame_idx))
            detection.write(f)
            f.write(struct.pack("f", score))

    @classmethod
    def load(cls, f):
        num = struct.unpack("i", f.read(4))[0]
        trajectory = cls()
        for _ in range(num):
            frame_idx = struct.unpack("i", f.read(4))[0]
            detection = Detection.load(f)
            score = struct.unpack("f", f.read(4))[0]
            trajectory.add(frame_idx, detection, score=score)
        return trajectory


def save_trajectories(trajectories, filename):
    with open(filename, "wb") as f:
        f.write(struct.pack("i", len(trajectories)))
        for trajectory in trajectories:
            trajectory.write(f)


def load_trajectories(filename):
    with open(filename, "rb") as f:
        num = struct.unpack("i", f.read(4))[0]
        trajectories = [Trajectory.load(f) for _ in range(num)]
    return trajectories


def compute_score(trajectory, frame_idx, detection):
    end = trajectory.end()
    latest = trajectory.latest()
    if frame_idx < end:
        return 0.0
    if frame_idx > end + 5:
        return 0.0
    delta = frame_idx - end
    score = detection.confidence
    score *= latest.iou(detection)
    score /= delta
    return score


class TrajectoryBuilder:
    def __init__(self, min_add_confidence=0.9, min_update_score=0.5):
        self.trajectories = []
        self.min_add_confidence = min_add_confidence
        self.min_update_score = min_update_score

    def add_detections(self, frame_idx, detections):
        used = [False for _ in detections]
        end_trajectories = []
        updated_trajectories = []
        for trajectory in self.trajectories:
            updated = False
            for di, detection in enumerate(detections):
                score = compute_score(trajectory, frame_idx, detection)
                if score > self.min_update_score:
                    new_trajectory = trajectory.add_copy(
                        frame_idx, detection, score=score
                    )
                    updated_trajectories.append(new_trajectory)
                    updated = True
                    used[di] = True
            if not updated:
                end_trajectories.append(trajectory.clone())
        unused_detections = [
            detection for d_used, detection in zip(used, detections) if not d_used
        ]
        new_trajectories = self.create_new_trajectories(frame_idx, unused_detections)
        self.trajectories = new_trajectories + end_trajectories + updated_trajectories

    def create_new_trajectories(self, frame_idx, detections):
        new_trajectories = []
        for detection in detections:
            if detection.confidence > self.min_add_confidence:
                new_traj = Trajectory()
                new_traj.add(frame_idx, detection, score=detection.confidence)
                new_trajectories.append(new_traj)
        return new_trajectories
