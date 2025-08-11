import json
import argparse
import datetime
from pathlib import Path
from collections import defaultdict
from moviepy.video.io.VideoFileClip import VideoFileClip
from tabulate import tabulate


def time_to_seconds(time_str):
    """Convert 'HH:MM:SS.mmm' to total seconds as a float."""
    h, m, s = time_str.split(':')
    seconds = float(s)
    return int(h) * 3600 + int(m) * 60 + seconds

def seconds_to_hms(seconds):
    """Convert seconds to 'HH:MM:SS.mmm' format."""
    td = datetime.timedelta(seconds=seconds)
    hms = str(td)
    if '.' in hms:
        hms, ms = hms.split('.')
        hms = f"{hms}.{ms[:3]}"  # Ensure milliseconds are in 3 digits
    else:
        hms += ".000"
    return hms

def save_clips(output_dir: Path, video_file: Path, intervals: list, keep_unlisted: bool):
    """Save video clips based on time intervals."""
    intervals_by_roi = defaultdict(list)

    with VideoFileClip(str(video_file)) as video:

        for idx, (defect_name, roi_id, start, end) in enumerate(intervals, start=1):

            # Group by ROI_ID for keep_unlisted purposes
            intervals_by_roi[roi_id].append((start, end))

            # Use subclipped method for your version
            clip = video.subclipped(start_time=start, end_time=end)

            # Save the clip to a file
            output_dir_tmp = output_dir / f"roi-{roi_id}" / defect_name if defect_name else output_dir
            output_dir_tmp.mkdir(parents=True, exist_ok=True)

            # Convert "HH:MM:SS.sss" to total seconds
            h, m, s = map(float, start.split(":"))
            start_seconds = h * 3600 + m * 60 + s

            # Compute the frame index (round to nearest frame)
            start_frame = round(start_seconds * 25)

            clip_path = output_dir_tmp / f"{video_file.stem}_{start_frame:05}.mp4"
            clip.write_videofile(
                str(clip_path),
                codec="libx264"
            )

        if keep_unlisted:

            intervals_unlisted = defaultdict(list)
            for roi_id, intervals_list in intervals_by_roi.items():
                # Step 1: Normalize and sort the intervals
                sorted_intervals = sorted((time_to_seconds(start), time_to_seconds(end)) for start, end in intervals_list)

                # Step 2: Merge overlapping intervals (just in case)
                merged = []
                for interval in sorted_intervals:
                    if not merged or interval[0] > merged[-1][1]:
                        merged.append(list(interval))
                    else:
                        merged[-1][1] = max(merged[-1][1], interval[1])

                # Step 3: Compute complement intervals (the "unlisted" ones)
                unlisted = []
                prev_end = 0.0
                for start, end in merged:
                    if start > prev_end:
                        unlisted.append((seconds_to_hms(prev_end), seconds_to_hms(start)))
                    prev_end = max(prev_end, end)

                duration = video.duration

                if prev_end < duration:
                    unlisted.append((seconds_to_hms(prev_end), seconds_to_hms(duration)))

                intervals_unlisted[roi_id] = unlisted

            idx = 1

            for roi_id, intervals_list in intervals_unlisted.items():
                for (start, end) in intervals_list:
                    # Use subclipped method for your version
                    clip = video.subclipped(start_time=start, end_time=end)

                    # Save the clip to a file
                    output_dir_tmp = output_dir / f"roi-{roi_id}" / "unlisted"
                    output_dir_tmp.mkdir(parents=True, exist_ok=True)

                    # Convert "HH:MM:SS.sss" to total seconds
                    h, m, s = map(float, start.split(":"))
                    start_seconds = h * 3600 + m * 60 + s

                    # Compute the frame index (round to nearest frame)
                    start_frame = round(start_seconds * 25)

                    clip_path = output_dir_tmp / f"{video_file.stem}_{start_frame:05}.mp4"
                    clip.write_videofile(
                        str(clip_path),
                        codec="libx264"
                    )

                    idx += 1

def make_parser():
    parser = argparse.ArgumentParser("Extract Clips from Videos")
    parser.add_argument("--datadir", type=Path, required=True, help="Root directory of the videos")
    parser.add_argument("--trim-start", type=str, required=False, help="Starting point of clip")
    parser.add_argument("--trim-end", type=str, required=False, help="Ending point of clip")
    parser.add_argument("--keep-unlisted", action="store_true", help="If set, retains video segments that are not listed")
    return parser

def get_roi_from_id(roi_json, id):
    for roi in roi_json:
        if roi["id"] == int(id):
            return (roi["top_left"][0], roi["top_left"][1], roi["bottom_right"][0], roi["bottom_right"][1])

def main(args: argparse.Namespace):

    root_path = Path(args.datadir)
    outdir = root_path.with_name(f"{root_path.name}-clips")

    # Define video extensions
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".ts"}

    # Recursively find video files
    video_files = [f for f in root_path.rglob("*") if f.suffix.lower() in video_extensions]

    for video_file in video_files:

        # Preserve relative path structure
        relative_path = video_file.relative_to(root_path).parent
        outdir_path = outdir / relative_path
        outdir_path.mkdir(parents=True, exist_ok=True)

        txt_file = video_file.with_suffix(".txt")
        json_file = video_file.with_suffix(".json")

        time_intervals = []
        if json_file.exists():
            # Has .json file
            with json_file.open() as f:
                json_content = json.load(f)

                for roi_str, defects_dict in json_content.items():
                    roi_id = roi_str.split(':')[-1]

                    for defect_str, intervals in defects_dict.items():
                        defect_name = defect_str.split(':')[-1]

                        for interval in intervals:
                            start, end = interval[0], interval[1]
                            time_intervals.append((defect_name, roi_id, start, end))

        elif txt_file.exists():
            # Has .txt file (eg: "00:00:01.340_00:00:04.110")
            with txt_file.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    start, end = line.strip().split("_")
                    time_intervals.append((None, None, start, end))

        else:
            if not (args.trim_start or args.trim_end):
                # Save original video to outdir
                with VideoFileClip(str(video_file)) as video:
                    video.write_videofile(
                        str(outdir_path),
                        fps=video.fps,
                        codec="libx264"
                    )
                continue

            # Has trim limits
            time_intervals.append((None, None, args.trim_start, args.trim_end))

        headers = ["Defect", "ROI", "Time Start", "Time End"]
        print(tabulate(time_intervals, headers=headers, tablefmt="grid"))

        # Process the video for the intervals
        save_clips(outdir_path, video_file, time_intervals, keep_unlisted=args.keep_unlisted)

        # Save segments between the clips
        # 1. Compute interval based on time_intervals and duration of the full video
        # 2. Call save_clips() with the computed time intervals


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)