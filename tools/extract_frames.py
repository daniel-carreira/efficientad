import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
import re
import json


def get_frame_start(filename):
    match = re.search(r'_(\d{5})\.\w+$', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))  # This returns the 5-digit clip ID
    return 0

def get_basename(basename):
    return re.sub(r'_\d{5}$', '', basename)

def extract_frames(video, interval, outdir, show_bar=True):
        
    capture = cv2.VideoCapture(video)
    capture.set(cv2.CAP_PROP_FPS, 0)
    nlen = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    basename = get_basename(video.stem)

    i = 0
    success, frame = capture.read()
    frame_count = get_frame_start(str(video))

    with tqdm(total=nlen,colour='YELLOW',desc = 'Extracting...',disable= not show_bar) as pbar:
        while success:
            if interval:
                if i % interval == 0:
                    frame_filename = outdir / f"{basename}_{frame_count:05d}.png"
                    cv2.imwrite(str(frame_filename), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            
            success, frame = capture.read()
            pbar.update(1)
            i+=1
            frame_count += 1

def make_parser():
    parser = argparse.ArgumentParser("Extract Image Dataset from Videos")

    parser.add_argument("--datadir", type=Path, required=True, help="directory of the videos")
    parser.add_argument("--interval", type=int, default=1, help="Space between frames")
    
    return parser
    

def main(args: argparse.Namespace):

    root_path = Path(args.datadir)

    # Define video extensions
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".ts"}

    # Recursively find video files
    video_files = [f for f in root_path.rglob("*") if f.suffix.lower() in video_extensions]

    for video_file in video_files:

        # Preserve relative path structure
        relative_path = video_file.relative_to(root_path.parent).parent
        outdir_path = Path("data/processed") / relative_path
        outdir_path.mkdir(parents=True, exist_ok=True)

        print(f"[Extractor] Processing clip: {video_file}")

        extract_frames(video=video_file, interval=args.interval, outdir=outdir_path, show_bar=True)


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)