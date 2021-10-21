import multiprocessing
from multiprocessing import Queue
import os
from pathlib import Path
from tqdm import tqdm
from .player_detector import PlayerDetector
from .detection import save_detections


class DetectProcess(multiprocessing.Process):
    def __init__(
        self,
        input_queue: Queue,
        output_queue: Queue,
        output_path: Path,
    ):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.output_path = Path(output_path)
        self.detector = PlayerDetector()

    def run(self):
        while True:
            item = self.input_queue.get()

            # Exit the process
            if item is None:
                break

            idx, frame_path = item
            output_path = self.output_path / f"{idx}.det"

            if output_path.exists():
                self.output_queue.put((idx, True))
                continue

            try:
                img_tensor = self.detector.image_to_tensor(frame_path)
                detections = self.detector.detect_people(img_tensor)
                save_detections(detections, output_path)
                created = True
            except Exception as e:  # pylint: disable=broad-except
                print(e)
                created = False
            self.output_queue.put((idx, created))


def detect_in_frames(
    frames_folder,
    output_path,
    num_workers=5,
):
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)

    input_queue = Queue()
    output_queue = Queue()

    processes = [
        DetectProcess(
            input_queue,
            output_queue,
            output_path,
        )
        for _ in range(num_workers)
    ]

    for proc in processes:
        proc.start()

    images = sorted([fn for fn in os.listdir(frames_folder) if fn.endswith(".png")])

    todo = 0
    for fn in tqdm(images):
        idx = fn.replace("f", "").replace(".png", "")
        input_queue.put((idx, os.path.join(frames_folder, fn)))
        todo += 1

    for _ in range(num_workers):
        input_queue.put(None)

    num_ok = 0
    for _ in tqdm(range(todo)):
        idx, ok = output_queue.get()
        if ok:
            num_ok += 1

    print(f"Detected {num_ok} / {todo}")

    for proc in processes:
        proc.terminate()

    return num_ok


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-W", "--num-workers", type=int, default=5)
    parser.add_argument("frames_folder")
    parser.add_argument("output_path")
    args = parser.parse_args()

    detect_in_frames(
        args.frames_folder,
        args.output_path,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
