import numpy as np


def build_extended_iou_matrix(det_bboxes, trk_bboxes, iou_matrix):
    """
    Creates an extended IoU matrix which includes detection and tracker ids.
    """
    # If no trackers are available, return an empty matrix
    if len(trk_bboxes) == 0:
        return np.array([])

    # Add ids of detections and trackers
    detection_ids = det_bboxes[:, -1]
    tracklet_ids = trk_bboxes[:, 4]

    M, N = iou_matrix.shape

    # Create a new matrix with additional rows/columns for ids
    new_matrix = np.zeros((M + 1, N + 1), dtype=object)
    new_matrix[1:, 1:] = (
        iou_matrix  # Place the original IoU matrix into the extended matrix
    )

    # Replace IoU values of 0 with "+"
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            if new_matrix[i, j] == 0:
                new_matrix[i, j] = "+"

    # Set the first row and column to hold the respective ids
    new_matrix[1:, 0] = detection_ids.astype(int)
    new_matrix[0, 1:] = tracklet_ids.astype(int)
    new_matrix[0, 0] = "D/T"

    return new_matrix

class TrackerLogger:
    def __init__(self, log_file):
        with open(log_file, 'w') as file:
            file.write(f"Log file initialized\n")
            self.log_file = log_file

    def add_log(self, text):
        with open(self.log_file, 'a') as file:
            file.write(text)
    
    def _separator(self):
        return "\n" + ("#" * 200) + "\n"

    def save_hungarian(self, matched, unmatched_dets, dets, trks, title = 'Hungarian Algorithm'):
       
        detection_header = f"\n-------------- {title} --------------\n"
        self.add_log(detection_header)
       
        _dets = dets[matched[:,0]]
        _dets_ids = _dets[:,-1]
        _trks = trks[matched[:,1]]  
        _trks_ids = _trks[:,6]

        M,N = matched.shape
        matched_new_matrix = np.zeros((M + 1, N), dtype=object)
        matched_new_matrix[1:,0] = _dets_ids.astype(int)
        matched_new_matrix[1:,1] = _trks_ids.astype(int)
        matched_new_matrix[0,0] = 'D'
        matched_new_matrix[0,1] = 'T'
        
        
        if unmatched_dets.size == 0:
            _unmatched_dets_ids = np.array([], dtype=int)
        else:
            _unmatched_dets = dets[unmatched_dets]
            _unmatched_dets_ids = _unmatched_dets[:,-1].astype(int)
        
        unmatched_dets_new_matrix = np.insert(_unmatched_dets_ids.astype(object), 0, "D")
        max_rows = max(len(matched_new_matrix), len(unmatched_dets_new_matrix))
        padded_array1 = np.pad(matched_new_matrix, ((0, max_rows - len(matched_new_matrix)), (0, 0)), constant_values="")
        padded_array2 = np.pad(unmatched_dets_new_matrix, (0, max_rows - len(unmatched_dets_new_matrix)), constant_values="")

        # Header
        self.add_log("M(D,T)\t\t\tUnmatched\n")
        self.add_log("---------------------------------------\n")

        # Data
        for row1, row2 in zip(padded_array1, padded_array2):
            row1_str = f"{row1[0]:<4} {row1[1]:<4}" if row1[0] != "" else "     "
            row2_str = f"{row2}" if row2 != "" else " "
            self.add_log(f"{row1_str}\t\t{row2_str}\n")
                
    
    def save_logs(self, n, bboxes, tracker_bboxes, iou_matrix):
        self.add_log(self._separator())        
        log_header = (
            f"\n-------------- Frame {n} --------------\n"
        )
        self.add_log(log_header)
        self._write_detections(bboxes)
        self._write_trackers(tracker_bboxes)
        self._write_iou_matrix(iou_matrix, title = 'Before Assignment-IOU Matrix (After tracker.predict())')


    def _write_detections(self, bboxes, title = 'Detections'):
      
        detection_header = f"\n-------------- {title} --------------\n"
        self.add_log(detection_header)

        for bbox in bboxes:
            x0, y0, x1, y1, score, class_idx, id = bbox

            line = f"id: {id}, x0: {x0:.3f}, y0: {y0:.3f}, x1: {x1:.3f}, y1: {y1:.3f}, score: {score:.3f}, class: {class_idx}\n"
            self.add_log(line)

    def _write_trackers(self, tracker_bboxes, title = 'Trackers'):

        tracker_header = f"\n-------------- {title} --------------\n"
        self.add_log(tracker_header)

        for tracker_bbox in tracker_bboxes:
            x0, y0, x1, y1, score, class_idx, id, age, hits, hit_streak, time_since_update, vx, vy, s = tracker_bbox

            line = (
                f"id: {int(id)}, x0: {x0:.3f}, y0: {y0:.3f}, x1: {x1:.3f}, y1: {y1:.3f}, score: {score:.2f}, class: {class_idx}"
                f"vx: {vx:.2f}, vy: {vy:.2f}, s: {s:.3f}, "
                f"age: {int(age)}, hits: {int(hits)}, hit_streak: {int(hit_streak)}, tsu: {int(time_since_update)}\n"
            )
            self.add_log(line)

    def _write_iou_matrix(self, iou_matrix,title = "IOU Matrix"):

        iou_header = f"\n-------------- {title} --------------\n"
        self.add_log(iou_header)

        for row in iou_matrix:
            formatted_row = [f"{value:<5.2f}" if isinstance(value, float) else f"{str(value):<5}" for value in row]
            line = "".join(formatted_row) + "\n"
            self.add_log(line)

    @staticmethod
    def build_extended_iou_matrix(dets, trks, iou_matrix):
        """
        Creates an extended IoU matrix which includes detection and tracker ids.
        """
        # If no trackers are available, return an empty matrix
        if len(trks) == 0:
            return np.array([])

        # Add ids of detections and trackers
        detection_ids = dets[:, -1]
        tracklet_ids = trks[:, 4]

        M, N = iou_matrix.shape

        # Create a new matrix with additional rows/columns for ids
        new_matrix = np.zeros((M + 1, N + 1), dtype=object)
        new_matrix[1:, 1:] = (
            iou_matrix  # Place the original IoU matrix into the extended matrix
        )

        # Replace IoU values of 0 with "+"
        for i in range(1, M + 1):
            for j in range(1, N + 1):
                if new_matrix[i, j] == 0:
                    new_matrix[i, j] = "+"

        # Set the first row and column to hold the respective ids
        new_matrix[1:, 0] = detection_ids.astype(int)
        new_matrix[0, 1:] = tracklet_ids.astype(int)
        new_matrix[0, 0] = "D/T"

        return new_matrix

