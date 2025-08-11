import os
import cv2
import json
import numpy as np
from shapely.geometry import Polygon
from skimage.draw import polygon as sk_polygon


def calculate_iou(bbox1, bbox2):
    # Calculate the (x1, y1) and (x2, y2) coordinates of the intersection rectangle
    x1_inter = max(bbox1[0], bbox2[0])
    y1_inter = max(bbox1[1], bbox2[1])
    x2_inter = min(bbox1[2], bbox2[2])
    y2_inter = min(bbox1[3], bbox2[3])

    # Calculate the area of intersection
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # Calculate the areas of the individual bounding boxes
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # Calculate union area
    union_area = bbox1_area + bbox2_area - inter_area

    # Calculate IoU
    iou = inter_area / union_area if union_area != 0 else 0
    return iou

def rasterize_polygon(polygon, crop_width, crop_height):
    # Extract the polygon's exterior coordinates
    coords = np.array(polygon.exterior.coords)
    x, y = coords[:, 0], coords[:, 1]

    # Generate the row and column indices for the polygon
    rr, cc = sk_polygon(y, x, shape=(crop_height, crop_width))

    # Create the annotation matrix
    annotation_matrix = np.zeros((crop_height, crop_width), dtype=np.uint8)
    annotation_matrix[rr, cc] = 1
    
    return annotation_matrix

def load_yolo_annotation_to_matrix(annotation_path, annotation_dims, roi_dims, class_name):
    """
    Load YOLO annotations and fill a matrix of size dims with ones where
    bounding boxes are located.
    
    Args:
        path (str): Path to the YOLO annotation text file.
        dims (tuple): Tuple of (width, height) for the image dimensions.
        
    Returns:
        np.ndarray: Matrix with ones in the area of the bounding boxes.
    """
    def clamp_value(value, lower_bound, upper_bound):
        return max(lower_bound, min(value, upper_bound))

    # Image width and height
    crop_width, crop_height, x_offset, y_offset = roi_dims

    image_width, image_height = annotation_dims
    
    # Initialize a matrix of zeros with shape (height, width)
    annotation_matrix = np.zeros((crop_height, crop_width), dtype=np.uint8)
    has_defect = False

    if class_name == 'normal':
        return annotation_matrix, False

    if not os.path.exists(annotation_path):
        return annotation_matrix, None
    
    # Open the annotation file
    with open(annotation_path, 'r') as file:
        for line in file:
            # Split the line into individual components
            parts = line.strip().split()

            # If labels are polygons
            if len(parts) > 5:
                # It's a polygon (YOLOv8 segmentation format)
                polygon_points = [float(p) for p in parts[1:]]  # Polygon points (x1, y1, ..., xn, yn)
                
                polygon_coords = [(polygon_points[i], polygon_points[i + 1]) 
                                  for i in range(0, len(polygon_points), 2)]
                
                # Convert normalized points to pixel coordinates
                polygon_pixels = [(int(x * image_width), int(y * image_height)) for x, y in polygon_coords]

                # Ensure polygon is closed (the last point is the same as the first)
                if polygon_pixels[0] != polygon_pixels[-1]:
                    polygon_pixels.append(polygon_pixels[0])

                # Offset polygon coordinates for cropping
                polygon = Polygon([(clamp_value(x - x_offset, 0, crop_width), clamp_value(y - y_offset, 0, crop_height)) for x, y in polygon_pixels])

                local_annotation_matrix = rasterize_polygon(polygon=polygon, crop_width=crop_width, crop_height=crop_height)

                annotation_matrix += local_annotation_matrix
                has_defect = True

            else:
                # Convert to appropriate types
                x_center = float(parts[1])  # x_center in normalized coordinates
                y_center = float(parts[2])  # y_center in normalized coordinates
                width = float(parts[3])  # width in normalized coordinates
                height = float(parts[4])  # height in normalized coordinates
                
                # Convert normalized values to pixel values
                x_center_pixel = int(x_center * image_width)
                y_center_pixel = int(y_center * image_height)
                width_pixel = int(width * image_width)
                height_pixel = int(height * image_height)
                
                # Calculate bounding box top-left and bottom-right coordinates
                xmin = max(0, x_center_pixel - width_pixel // 2)
                ymin = max(0, y_center_pixel - height_pixel // 2)
                xmax = min(image_width, x_center_pixel + width_pixel // 2)
                ymax = min(image_height, y_center_pixel + height_pixel // 2)
                
                # Adjust coordinates based on the offset (to map to the cropped region)
                xmin_crop = max(0, xmin - x_offset)
                ymin_crop = max(0, ymin - y_offset)
                xmax_crop = min(crop_width, xmax - x_offset)
                ymax_crop = min(crop_height, ymax - y_offset)

                # Fill the crop matrix with ones inside the bounding box area
                if xmax_crop > xmin_crop and ymax_crop > ymin_crop:
                    annotation_matrix[ymin_crop:ymax_crop, xmin_crop:xmax_crop] = 1
                    has_defect = True

    return annotation_matrix, has_defect

def load_roi(roi_path):
    if not roi_path:
        return None, None
    
    with open(roi_path, 'r') as file:
        roi = json.load(file)

    roi_info = roi["areas"][0]
    roi_type = roi_info["type"]
    roi = roi_info[roi_type]

    return roi_type, roi

def compute_roi_mask(roi_type, roi, shape):
    # Initialize an empty mask
    roi_zone = np.zeros(shape, dtype=np.uint8)

    if roi_type == 'bbox':
        # Extract coordinates for bbox
        xmin, ymin, xmax, ymax = roi
        xmin, ymin = max(0, xmin), max(0, ymin)  # Ensure values are within bounds
        xmax, ymax = min(shape[1], xmax), min(shape[0], ymax)  # Ensure values are within bounds
        
        # Fill the bbox area with ones
        roi_zone[ymin:ymax, xmin:xmax] = 1

    elif roi_type == 'polygon':
        # Convert polygon points to a NumPy array
        polygon = np.array(roi, dtype=np.int32)
        
        # Draw the polygon on the mask with value 1
        cv2.fillPoly(roi_zone, [polygon], 1)

    return roi_zone

def best_match_in_annotations(annotations, bbox):
    best_iou = 0
    best_match = None
    
    # Loop over each annotation and compute IoU
    for section in annotations['areas']:
        bbox_annotation = section['bbox']

        iou = calculate_iou(bbox_annotation, bbox)
        
        # Keep track of the highest IoU
        if iou > best_iou:
            best_iou = iou
            best_match = section
    
    return best_match

def export_annotations(dims, dets, output_path, format='yolo'):
    width, height = dims
    if format == 'yolo':
        # Format: {cls_id} {x_center} {y_center} {width} {height}
        with open(output_path, mode='w') as f:
            for bbox in dets:
                x1, y1, x2, y2 = bbox

                # Compute YOLO coordinates (normalized)
                x_center = (x1 + x2) / 2 / width
                y_center = (y1 + y2) / 2 / height
                bbox_width = (x2 - x1) / width
                bbox_height = (y2 - y1) / height

                # Write default class [0 - Unknown] and normalized bbox in YOLO format
                f.write(f"0 {x_center:.4f} {y_center:.4f} {bbox_width:.4f} {bbox_height:.4f}\n")