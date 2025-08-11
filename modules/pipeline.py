import os
import cv2
import json
import argparse
import numpy as np
import torch
import torch_tensorrt
import torch.nn.functional as F
from typing import List, Tuple, Iterator

from tracker import Sort
from helpers.player_helper import create_player


class DefaultPipeline:

    def __init__(self, screen_size=(1920, 1080)):
        self.id = 0
        self.sources = {}
        self.screen_size = screen_size
    
    # ============================ INITIALIZATION ============================
    def get_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, required=True)
        parser.add_argument('--dest', type=str, default="results/inference", required=False)
        parser.add_argument('--with-vis', action="store_true")
        parser.add_argument('--with-rec', action="store_true")

        parser.add_argument('--cam0', type=str, required=True)

        return parser
    
    def parse_args(self) -> dict:
        self.args = self.get_parser().parse_args()

        # Open config file
        with open(self.args.config) as f:
            self.args.config = json.load(f)

        return self.args
    
    def get_frame_iter(self) -> Iterator[Tuple[Tuple[int, np.ndarray, int], ...]]:
        def make_generator(source_id, player):
            return ((source_id, frame, frame_idx) for frame, frame_idx in player)

        players = [
            make_generator(source_id, source["player"])
            for source_id, source in self.sources.items()
        ]
        return zip(*players)

    def prepare(self, device) -> None:
        assert self.args is not None, "Run parse_args() to populate args."

        self.broker = None

        # Load models
        optional_keys = ["combined_map_threshold"]
        for source_config in self.args.config["sources"]:

            source_id = source_config["id"]
            cam_param = getattr(self.args, f"cam{source_id}", None)
            if cam_param is None:
                continue
            
            # Add source
            self.sources[source_id] = {
                "player": create_player(source_id=source_id, source=cam_param, verbose=True),
                "trks": Sort(
                    max_age_infant=self.args.config["params"]["tracker_max_age"],
                    max_age_mature=self.args.config["params"]["tracker_max_age"],
                    min_hits=self.args.config["params"]["tracker_min_hits"],
                    iou_threshold=self.args.config["params"]["tracker_iou_threshold"]),
                "trks_seen": set()
            }

            for roi_str, models_config in source_config["rois"].items():

                # Extract ROI dims
                roi_values = roi_str.split(":")
                width, height, x_offset, y_offset = map(int, roi_values)

                source_config["rois"][roi_str] = ((width, height, x_offset, y_offset), source_config["rois"][roi_str])

                for i, model_config in enumerate(models_config):

                    model = torch.jit.load(model_config["path"])
                    model = model.to(device).eval()

                    value = {
                        "id": model_config["id"],
                        "model": model,
                        "roi": (width, height, x_offset, y_offset),
                        "input_size": model_config["input_size"]
                    }
                    params = model_config.get("params", {})

                    for optional_key in optional_keys:
                        if optional_key in params:
                            value[optional_key] = params[optional_key]

                    models_config[i] = value

        # Config writers
        self.writer = None

        if self.args.with_rec:
            output = self.args.dest
            os.makedirs(output, exist_ok=True)

            any_source = next(iter(self.sources.values()))
            outpath = os.path.join(output, f"{any_source['player'].base_name}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            self.writer = cv2.VideoWriter(outpath, fourcc, any_source["player"].fps, self.screen_size)

            if not self.writer.isOpened():
                raise RuntimeError("VideoWriter failed to open.")

        return self.args
    
    # ============================ MAIN ============================
    def postprocess_ad(self, map_frame: np.ndarray, map_roi: np.ndarray, config: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Postprocesses the output of an anomaly detection model.

        Args:
            map_frame (np.ndarray): Frame-wise anomaly map,
            map_roi (np.ndarray): ROI-wise anomaly map,
            config (dict): Configuration dictionary containing settings:
                {
                    'roi': (width, height, x0, y0),         # Region of interest in the original image
                    'input_size': (width, height),          # Model input resolution
                    'combined_map_threshold': float,        # Anomaly score threshold (>0)
                }

        Returns:
            list[tuple]: A list of detected objects.
                Each tuple contains:
                    - bbox (list[float]): Bounding box [x0, y0, x1, y1]
                    - score (float): Detection anomaly score
                    - label (int): Predicted class index
        """
        width, height, x0, y0 = config["roi"]

        # Interpolate on GPU
        anomaly_map = F.interpolate(map_roi, size=(height, width), mode='bilinear', align_corners=False)

        # Apply thresholding on GPU
        threshold = config['combined_map_threshold']
        anomaly_map = torch.where(anomaly_map >= threshold, anomaly_map, torch.tensor(0.0, device=anomaly_map.device))

        # Optionally move to CPU if needed for OpenCV
        anomaly_map = anomaly_map[0, 0].detach().cpu().numpy()

        # Calculate padding in resized dimensions
        s_w = width / 64
        s_h = height / 64

        pad_w = int(4 * s_w)
        pad_h = int(4 * s_h)

        # Crop padding from resized maps (local coords)
        anomaly_map = anomaly_map[pad_h:height - pad_h, pad_w:width - pad_w]

        # Update where the map will go in the full-size frame
        x0 += pad_w
        y0 += pad_h
        width -= 2 * pad_w
        height -= 2 * pad_h

        # Place ROI-wise anomaly map into the Frame-wise anomaly map
        map_frame[y0:y0+height, x0:x0+width] = anomaly_map

        # Place Frame-wise anomaly map into the ROI-wise anomaly map
        anomaly_map = map_frame[y0:y0+height, x0:x0+width]

        # Compute mask of the valid anomaly pixels (anomaly_score > 0)
        mask = (anomaly_map > 0).astype(np.uint8)

        bboxes, scores = [], []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            score = np.max(anomaly_map[y:y+h, x:x+w])

            bbox = (x+x0, y+y0, (x+w)+x0, (y+h)+y0)

            bboxes.append(bbox)
            scores.append(score)

        classes = [0] * len(bboxes)
        return bboxes, scores, classes

    def process_trackers(self, source_id: int, dets: np.ndarray) -> np.ndarray:
        trks = self.sources[source_id]["trks"].update(dets, vx=0, vy=0)
        return trks

   # ============================ DRAWING ============================
    def draw_info(self, frame: np.ndarray, desc: str = "") -> np.ndarray:
        frame = frame.copy()

        # Draw params
        font_scale = 0.7
        font_color = (255, 255, 255)
        thickness = 1

        # Write frame info
        cv2.putText(frame, desc, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness)

        return frame

    def draw_bbox_annotation_generic(frame: np.ndarray, bbox: List[int], desc: str, with_bbox=False, color=(0, 0, 255)) -> None:
        # Text properties
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.5
        font_thickness = 1

        # Define text color (foreground) and background color
        fg_color = (255, 255, 255)  # White
        bg_color = color

        # Prepare text with its position and size
        text_size, baseline = cv2.getTextSize(desc, font, font_scale, font_thickness)
        text_width, text_height = text_size
        label_position = (int(bbox[0]), int(bbox[1]) - 5) 

        bg_top_left = (label_position[0], label_position[1] - text_height - baseline)
        bg_bottom_right = (label_position[0] + text_width, label_position[1] + baseline)
        cv2.rectangle(frame, bg_top_left, bg_bottom_right, bg_color, thickness=-1)  

        if desc:
            cv2.putText(frame, desc, label_position, font, font_scale, fg_color, font_thickness)

        if with_bbox:
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), bg_color, 2)

    def draw_graphics(self, frame: np.ndarray, trks: np.ndarray, anomaly_map: np.ndarray = None) -> np.ndarray:
        overlay = frame.copy()
        if anomaly_map is not None:
            # Normalize the anomaly map to range [0, 255]
            anomaly_map = np.where(anomaly_map > 1, 1, anomaly_map)
            anomaly_map = anomaly_map * 255

            # Binary mask scaled to 0 or 255
            alpha = (anomaly_map > 0).astype(np.uint8) * 255
            alpha = cv2.merge([alpha, alpha, alpha])

            # Apply a color map to the anomaly map
            heatmap = cv2.applyColorMap(anomaly_map.astype(np.uint8), cv2.COLORMAP_JET)

            # =============== DRAWING =============== #
            # ===> Draw Anomaly Map
            cv2.addWeighted(frame, 1, heatmap, 0.5, 0, dst=overlay)
            overlay[alpha == 0] = frame[alpha == 0]

        for trk in trks:
            # Format bbox
            bbox = trk[:4].astype(int)

            # Format description
            desc = f'{trk[4]:.2f}'

            # Draw trks
            self.draw_bbox_annotation_generic(overlay, bbox, desc=desc, with_bbox=True, color=(0, 192, 0))

        return overlay

    def draw_combined_graphics_generic(frames: List[np.ndarray], layout: Tuple[int, int], screen_size: Tuple[int, int] = (2560, 1440)) -> np.ndarray:
        screen_w, screen_h = screen_size
        rows, cols = layout
        n = len(frames)

        assert 1 <= n <= rows * cols, "Too few or too many frames for the given layout"

        # Step 1: Measure max widths per column and max heights per row
        max_col_widths = [0] * cols
        max_row_heights = [0] * rows

        for idx, frame in enumerate(frames):
            row = idx // cols
            col = idx % cols
            h, w = frame.shape[:2]
            max_col_widths[col] = max(max_col_widths[col], w)
            max_row_heights[row] = max(max_row_heights[row], h)

        # Step 2: Normalize column widths and row heights to screen size
        total_width = sum(max_col_widths)
        total_height = sum(max_row_heights)

        col_scales = [screen_w * (w / total_width) for w in max_col_widths]
        row_scales = [screen_h * (h / total_height) for h in max_row_heights]

        # Step 3: Compute cumulative offsets
        col_offsets = [int(sum(col_scales[:i])) for i in range(cols)]
        row_offsets = [int(sum(row_scales[:i])) for i in range(rows)]

        combined = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

        for idx, frame in enumerate(frames):
            row = idx // cols
            col = idx % cols

            h, w = frame.shape[:2]
            cell_w = int(col_scales[col])
            cell_h = int(row_scales[row])
            scale = min(cell_w / w, cell_h / h)

            resized_w, resized_h = int(w * scale), int(h * scale)
            resized_frame = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

            x_offset = col_offsets[col] + (cell_w - resized_w) // 2
            y_offset = row_offsets[row] + (cell_h - resized_h) // 2

            combined[y_offset:y_offset + resized_h, x_offset:x_offset + resized_w] = resized_frame

        return combined

    def draw_combined_graphics(self, frames: List[np.ndarray], desc: str = "") -> np.ndarray:
        frame_combined = self.draw_combined_graphics_generic(frames=frames, layout=(1, len(frames)), screen_size=self.screen_size)
        frame_combined_with_info = self.draw_info(frame=frame_combined, desc=desc)
        return frame_combined_with_info
    
    def view(self, window_name: str, frame: np.ndarray) -> None:
        # Show in fullscreen
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(window_name, frame)
    
    def write(self, frame: np.ndarray):
        if self.writer:
            self.writer.write(frame)
    
    # ============================ EXITING ============================
    def release(self):
        if self.writer:
            self.writer.release()

        for source in self.sources.values():
            source["player"].release()