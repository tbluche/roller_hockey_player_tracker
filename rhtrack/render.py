from collections import defaultdict
from pathlib import Path
import subprocess
import matplotlib.pyplot as plt
import matplotlib.patches
import numpy as np
from tqdm import tqdm
import cv2
from .trajectories import TrajectoryBuilder, save_trajectories, load_trajectories


def create_box(trajectories):
    start_frame = max([trajectory.start() for trajectory in trajectories])
    end_frame = min([trajectory.end() for trajectory in trajectories])
    cropped_trajectories = [
        trajectory.crop(start_frame, end_frame) for trajectory in trajectories
    ]
    assert all(
        [
            len(trajectory) == len(cropped_trajectories[0])
            for trajectory in cropped_trajectories
        ]
    )
    frame_to_box = {}
    for ts in zip(*cropped_trajectories):
        frame_ids, detections = zip(*ts)
        frame_idx = frame_ids[0]
        assert all([f_idx == frame_idx for f_idx in frame_ids])
        frame_to_box[frame_idx] = np.array(
            [list(detection.middle()) for detection in detections]
        )
    return frame_to_box


def render(clip_path, box_players=None):
    frames = sorted(list((clip_path / "frames").glob("f*.png")))
    print("Read trajectories")
    trajectories = load_trajectories(clip_path / "trajectories" / "trajectories")
    players = [
        line.strip().split(":", 1)
        for line in (clip_path / "trajectories" / "players.txt").open()
        if ":" in line.strip()
    ]
    player_to_trajectories = defaultdict(list)
    for tid_str, player_name in players:
        if player_name.strip() != "":
            player_to_trajectories[player_name.strip()].append(
                trajectories[int(tid_str)]
            )

    print("Players")
    print(list(player_to_trajectories.keys()))
    print("Merge player trajectories")
    player_to_trajectory = {}
    for player_name, player_trajectories in player_to_trajectories.items():
        sorted_player_trajectories = sorted(
            player_trajectories, key=lambda x: x.start()
        )
        new_trajectory = sorted_player_trajectories[0]
        if len(sorted_player_trajectories) > 1:
            for next_trajectory in sorted_player_trajectories[1:]:
                if next_trajectory.start() >= new_trajectory.end():
                    new_trajectory = new_trajectory.merge(next_trajectory)
        player_to_trajectory[player_name] = new_trajectory.interpolate()

    box = {}
    if box_players is not None:
        box = create_box(
            [player_to_trajectory[player_name] for player_name in box_players]
        )
    print("Write output frames")
    video_frames = clip_path / "output_frames"
    video_frames.mkdir()

    for fi, frame in enumerate(tqdm(frames)):
        img = cv2.imread(str(frame))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(15, 10))
        plt.imshow(img)
        ax = plt.gca()
        cmap = plt.cm.jet
        for li, (label, trajectory) in enumerate(player_to_trajectory.items()):
            color = cmap(li / len(player_to_trajectory))
            if trajectory.start() > fi:
                continue
            trajectory.plot(max_idx=fi, color=color, label=label)
        if fi in box:
            mpl_box = matplotlib.patches.Polygon(
                box[fi], color="r", alpha=0.3, fill=True
            )
            ax.add_patch(mpl_box)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.savefig(str(video_frames / f"f{fi + 1:07d}.png"), bbox_inches="tight")
        plt.close()

    print("Render video")
    output_video = clip_path / "output.mp4"
    cmd = [
        "ffmpeg",
        "-i",
        str(video_frames / "f%07d.png"),
        "-c:v",
        "libx264",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        str(output_video),
    ]
    subprocess.check_call(cmd)
    print(f"Video written to {output_video}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Render video clip with player tracking")
    parser.add_argument("-p", "--project", type=str, required=True, help="Project name")
    parser.add_argument("--show-box", type=str, help="Comma-separated list of players")
    parser.add_argument("clip_name")
    args = parser.parse_args()
    project_path = Path(args.project)
    clip_path = project_path / args.clip_name
    box_players = None
    if args.show_box is not None:
        box_players = args.show_box.split(",")
    render(clip_path, box_players=box_players)
