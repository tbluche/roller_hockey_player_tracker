from pathlib import Path
import youtube_dl

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Create a project from a youtube video")
    parser.add_argument("youtube_url")
    parser.add_argument("project_name")
    args = parser.parse_args()
    output_path = Path(args.project_name)
    output_path.mkdir()

    ydl_opts = {"format": "mp4", "outtmpl": str(output_path / "match.mp4")}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([args.youtube_url])
