import cv2
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import queue
import threading


# ================== Factory function ==================
def create_player(source, buffer_size = None, output=None, verbose=True):
    if isinstance(source, (str, Path)) and Path(source).is_dir():
        return ImagePlayer(source, output=output, verbose=verbose)
    else:
        if buffer_size is None:
            return VideoPlayer(source, output=output, verbose=verbose)
        
        return VideoPlayerBuffered(source, buffer_size=buffer_size, outdir=output, verbose=verbose)


class Player:
    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError

    def write(self, frame):
        raise NotImplementedError

    def release(self):
        pass

class ImagePlayer(Player):
    def __init__(self, folder, output=None, verbose=True):
        self.image_paths = sorted([p for p in Path(folder).rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
        if not self.image_paths:
            raise RuntimeError(f"No images found in: {folder}")

        self.total = len(self.image_paths)
        self.folder = Path(folder)
        self.verbose = verbose
        self.frame_idx = 0
        self.base_name = self.folder.name
        self._pbar = tqdm(total=self.total, desc="[Image] Processing", unit="frame") if self.verbose else None
        self.mode = 'image'

        if output is not None:
            self.out_dir = Path(output)
            self.out_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.out_dir = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.frame_idx >= self.total:
            self.release()
            raise StopIteration

        image_path = self.image_paths[self.frame_idx]
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        idx = self.frame_idx
        self.frame_idx += 1
        if self._pbar:
            self._pbar.update(1)
        return frame, idx

    def write(self, frame):
        if self.out_dir:
            name = self.image_paths[self.frame_idx - 1].name
            out_path = self.out_dir / name
            cv2.imwrite(str(out_path), frame)

            return out_path

    def release(self):
        if self._pbar:
            self._pbar.close()

class VideoPlayer:
    def __init__(self, source, output=None, verbose=True):
        if source.startswith("rtsp://"):
            # ================== GSTREAMER (1.20.3) ==================
            # RTSP Stream on CPU
            pipeline = (
                f"rtspsrc location={source} latency=0 ! queue ! "
                "rtph264depay ! queue ! h264parse ! queue ! "
                "avdec_h264 ! queue ! "
                "videoconvert ! appsink caps=\"video/x-raw,format=BGR\" sync=false"
            )
            # RTSP Stream on GPU
            # pipeline = (
            #     f"rtspsrc location={source} ! queue ! "
            #     "rtph264depay ! queue ! h264parse ! queue ! "
            #     "nvh264dec ! queue ! "
            #     "nvvideoconvert ! appsink caps=\"video/x-raw,format=BGR\" sync=false"
            # )
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        else:
            # ================== GSTREAMER (1.20.3) ==================
            # VIDEO on CPU
            pipeline = (
                f"filesrc location={source} latency=0 ! queue ! "
                "tsdemux ! queue ! h264parse ! queue ! "
                "avdec_h264 ! queue !"
                "videoconvert ! appsink caps=\"video/x-raw,format=BGR\" sync=false"
            )
            # VIDEO on GPU
            # pipeline = (
            #     f"filesrc location={source} ! queue ! "
            #     "tsdemux ! queue ! h264parse ! queue ! "
            #     "nvh264dec ! queue !"
            #     "nvvideoconvert ! appsink caps=\"video/x-raw,format=BGR\" sync=false"
            # )
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

            # ==================       FFMPEG       =================
            # VIDEO on CPU
            #self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)

        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {source}")

        self.is_camera = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) <= 0
        self.total = None if self.is_camera else int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps, self.size = self.get_video_properties()
        self.verbose = verbose
        self.frame_idx = 0

        if isinstance(source, (str, Path)) and Path(source).is_file():
            self.base_name = Path(source).stem
        else:
            self.base_name = 'rtsp'

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.base_name = f"{self.base_name}_{timestamp}"

        self._pbar = tqdm(total=self.total, desc="[Video] Processing", unit="frame") if self.verbose and self.total else None

        if output is not None:
            Path(output).mkdir(parents=True, exist_ok=True)
            self.out_path = Path(output) / f"{self.base_name}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(str(self.out_path), fourcc, self.fps, self.size)
        else:
            self.writer = None

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret or self.frame_idx == self.total:
            self.release()
            raise StopIteration

        idx = self.frame_idx
        self.frame_idx += 1
        if self._pbar:
            self._pbar.update(1)
        return frame, idx

    def write(self, frame):
        if self.writer:
            self.writer.write(frame)

    def get_video_properties(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        size = (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        return fps, size

    def release(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'writer') and self.writer:
            self.writer.release()
        if self._pbar:
            self._pbar.close()

class VideoPlayerBuffered:
    def __init__(self, source: str, buffer_size: int, outdir: str = None, verbose: bool = False):
        self.player = VideoPlayer(source, output=outdir, verbose=False)
        self.size = self.player.size
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.stop_flag = threading.Event()
        self.peak_buffer_size = 0

        self._pbar = tqdm(total=self.player.total, desc="[Video] Processing", unit="frame") if verbose and self.player.total else None
        self.thread = threading.Thread(target=self._fill_buffer, daemon=True)
        self.thread.start()

    def _fill_buffer(self):
        for frame in self.player:
            if self.stop_flag.is_set():
                break
            try:
                self.buffer.put(frame, timeout=0.1)  # block if full
                self.peak_buffer_size = max(self.peak_buffer_size, self.buffer.qsize())
            except queue.Full:
                continue  # buffer full, try again

        self.stop_flag.set()

    def __iter__(self):
        return self

    def __getattr__(self, name):
        return getattr(self.player, name)

    def __next__(self):
        while not self.stop_flag.is_set() or not self.buffer.empty():
            try:
                frame = self.buffer.get(timeout=0.1)  # wait for frame
                if self._pbar:
                    self._pbar.update(1)
                return frame
            except queue.Empty:
                continue  # wait for frame

        raise StopIteration

    def release(self):
        self.stop_flag.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.player.release()
        if self._pbar:
            self._pbar.close()