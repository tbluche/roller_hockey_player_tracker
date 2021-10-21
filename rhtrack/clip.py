from pathlib import Path
import subprocess
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from .detect import detect_in_frames
from .detection import load_detections
from .trajectories import TrajectoryBuilder, save_trajectories


def plot_trajectory_helper(trajectory, frames_path, output_file):
    start, _ = trajectory[0]
    middle, _ = trajectory[len(trajectory) // 2]
    end, _ = trajectory[-1]

    plt.figure(figsize=(15, 10))
    for ti, fi in enumerate([start, middle, end]):
        plt.subplot(1, 3, ti + 1)
        img = cv2.imread(str(frames_path / f"f{fi + 1:07d}.png"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        trajectory.plot(max_idx=fi)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.xlabel(f"Frame {fi}", fontsize=16)
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


def create_clip(project_path, clip_name, start, duration, num_workers=5):
    print("Extract clip")
    output_path = project_path / clip_name
    output_path.mkdir()
    cmd = [
        "ffmpeg",
        "-ss",
        start,
        "-i",
        str(project_path / "match.mp4"),
        "-t",
        f"00:00:{duration:02d}",
        "-c",
        "copy",
        str(output_path / "clip.mp4"),
    ]
    subprocess.check_call(cmd)
    print("Extract frames")
    frames_path = output_path / "frames"
    detections_path = output_path / "detections"
    frames_path.mkdir()
    detections_path.mkdir()
    cmd = [
        "ffmpeg",
        "-i",
        str(output_path / "clip.mp4"),
        str(frames_path / "f%07d.png"),
    ]
    subprocess.check_call(cmd)
    print("Detect players")
    num_frames = detect_in_frames(
        frames_path,
        detections_path,
        num_workers=num_workers,
    )
    print("Find player trajectories")
    trajectories_path = output_path / "trajectories"
    trajectories_path.mkdir()
    builder = TrajectoryBuilder()
    for fi in tqdm(range(num_frames)):
        detections = load_detections(detections_path / f"{fi + 1:07d}.det")
        builder.add_detections(fi, detections)
    print(f"Saving {len(builder.trajectories)} trajectories to {trajectories_path}")
    save_trajectories(builder.trajectories, trajectories_path / "trajectories")
    with (trajectories_path / "players.txt").open("w") as f:
        for ti, trajectory in enumerate(tqdm(builder.trajectories)):
            f.write(f"{ti}: player_name\n")
            plot_trajectory_helper(
                trajectory, frames_path, trajectories_path / f"{ti}.png"
            )
    print(f"Edit {trajectories_path}/players.txt and run the next step")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Create a clip for a project")
    parser.add_argument(
        "-s",
        "--start",
        type=str,
        default="00:00:00",
        help="Start time (format hh:mm:ss)",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        default=10,
        help="Duration in seconds (max 60; default: 10)",
    )
    parser.add_argument(
        "-W",
        "--num-workers",
        type=int,
        default=5,
        help="Number of workers for player detection",
    )
    parser.add_argument("-p", "--project", type=str, required=True, help="Project name")
    parser.add_argument("clip_name")
    args = parser.parse_args()
    project_path = Path(args.project)
    create_clip(
        project_path,
        args.clip_name,
        args.start,
        args.duration,
        num_workers=args.num_workers,
    )
