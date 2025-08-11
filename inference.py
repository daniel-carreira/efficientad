#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch_tensorrt
from typing import Tuple

from modules.pipeline import DefaultPipeline


def process_frame(source_id: int, source_frame: np.ndarray, pipeline: DefaultPipeline, device: torch.device) -> Tuple[bool, np.ndarray, np.ndarray]:
    source = pipeline.args.config["sources"][source_id]
    dets = {
        "bboxes": [],
        "scores": [],
        "classes": []
    }
    shape = source_frame.shape[:2]
    map_combined = np.zeros(shape, dtype=np.float32)

    input_tensor = torch.from_numpy(source_frame) # BGR
    input_tensor = input_tensor.to(device) # C, H, W

    for roi, models in source["rois"].values():
        width, height, x0, y0 = roi

        # =================== Pre-Process ===================
        input_tensor_cropped = input_tensor[y0:y0+height, x0:x0+width, :]
        input_tensor_cropped = input_tensor_cropped.permute(2, 0, 1).float()
        input_tensor_cropped = input_tensor_cropped.unsqueeze(0)
        
        for model_config in models:
            input_tensor_cropped_resized = F.interpolate(input_tensor_cropped, size=model_config["input_size"], mode='bilinear', align_corners=False, antialias=True)

            # BGR -> RGB
            input_tensor_cropped_resized = input_tensor_cropped_resized[:, [2, 1, 0], :, :]
            input_tensor_cropped_resized.div_(255.0)

            # =================== Inference ===================
            outputs = model_config["model"](input_tensor_cropped_resized)

            # =================== Post-Process ===================
            bboxes, scores, classes = pipeline.postprocess_ad(map_frame=map_combined, map_roi=outputs, config=model_config)

            # Add bboxes, scores and classes to frame list
            dets["bboxes"].extend(bboxes)
            dets["scores"].extend(scores)
            dets["classes"].extend(classes)

    # ================= Trackers =================
    trks = pipeline.process_trackers(source_id=source_id, dets=(dets["bboxes"], dets["scores"], dets["classes"]))

    return True, trks, map_combined

def main(pipeline: DefaultPipeline):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare all components, and load models to device
    args = pipeline.prepare(device=device)
    
    for frame_sources in pipeline.get_frame_iter():

        frames_draw = []

        for source_id, frame_source, frame_source_count in frame_sources:

            success, trks, anomaly_map = process_frame(
                source_id=source_id, 
                source_frame=frame_source,
                pipeline=pipeline,
                device=device)
        
            if args.with_rec or args.with_vis:
                frames_draw.append(pipeline.draw_graphics(frame=frame_source, trks=trks, anomaly_map=anomaly_map))

        if args.with_rec or args.with_vis:
            frame_combined = pipeline.draw_combined_graphics(frames=frames_draw, desc=f'{frame_source_count}')

            if args.with_rec:
                pipeline.write(frame_combined)

            if args.with_vis:
                pipeline.view(window_name="EfficientAD", frame=frame_combined)
        
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

if __name__ == '__main__':

    pipeline = DefaultPipeline(screen_size=(1920, 1080))
    pipeline.parse_args()

    try:
        main(pipeline)

    except Exception as e:
        raise e