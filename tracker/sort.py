"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import print_function
import numpy as np
from typing import List
from filterpy.kalman import KalmanFilter

from .logger import TrackerLogger
from .utils import iou_batch, convert_bbox_to_z, convert_x_to_bbox, associate_detections_to_trackers

np.random.seed(0)


class KalmanBoxTracker(object):
    """
    Kalman filter-based tracker that uses bounding box information to track objects across frames.
    """
    # -> Singleton counter
    tracker_counter = 0


    def __init__(self, bbox, vx=0, vy=0):
        """
        Initialize the tracker with an initial bounding box, and optional velocity in x and y.
        """
        KalmanBoxTracker.tracker_counter += 1

        # Init params
        self.id = KalmanBoxTracker.tracker_counter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.score = bbox[4]
        self.class_idx = bbox[5]

        # Init Kalman filter matrices
        self._initialize_kf(bbox)

        # Adjust initial velocity
        self.kf.x[4][0] = vx
        self.kf.x[5][0] = vy

        # Saving history of tracker for debug reasons
        self.history = []
        self.state = np.array([bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], self.id, 0, 0, 0, 0, vx, vy, 0], dtype=np.float32)


    def _initialize_kf(self, bbox):
        """Initialize the Kalman filter matrices."""
        # Define the constant velocity model
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                               [0, 1, 0, 0, 0, 1, 0],
                               [0, 0, 1, 0, 0, 0, 1],
                               [0, 0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 0, 1]])  # State transition matrix

        # Measurement matrix
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0, 0]])

        # Measurement noise (adjusted for position and scale)
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01  # Process noise
        self.kf.Q[4:, 4:] *= 0.01

        # Initialize the filter's state vector with the bounding box
        self.kf.x[:4] = convert_bbox_to_z(bbox[:4])


    def update(self, bbox):
        """
        Updates the tracker with the new bounding box.
        """
        self.kf.update(convert_bbox_to_z(bbox[:4]))
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.score = bbox[4]
        self.class_idx = bbox[5]
        self.history = []

        self.state[:6] = bbox[:6]
        self.state[8] = self.hits
        self.state[9] = self.hit_streak
        self.state[11] = self.kf.x[4][0]
        self.state[12] = self.kf.x[5][0]
        self.state[13] = self.kf.x[6][0]


    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()
        bbox = convert_x_to_bbox(self.kf.x)
        self.history.append(bbox)
        self.age += 1
        self.time_since_update += 1
        self.hit_streak = 0 if self.time_since_update > 1 else self.hit_streak

        self.state[:4] = bbox
        self.state[7] = self.age
        self.state[9] = self.hit_streak
        self.state[10] = self.time_since_update
        self.state[11] = self.kf.x[4][0]
        self.state[12] = self.kf.x[5][0]
        self.state[13] = self.kf.x[6][0]

        return bbox
    

    def obj(self):
        """
        Returns the object's state
        """
        pos = convert_x_to_bbox(self.kf.x)[0]
        vx = self.kf.x[4][0]
        vy = self.kf.x[5][0]
        s = self.kf.x[6][0]

        return np.array([
            pos[0], pos[1], pos[2], pos[3],
            self.score,
            self.class_idx,
            self.id,
            self.age,
            self.hits,
            self.hit_streak,
            self.time_since_update,
            vx, vy, s
        ], dtype=np.float32)
    

class Sort(object):
    """
    The state of each target is modelled as:
    x = [u, v, s, r, u,˙ v,˙ s˙]T
    
    Where u and v represent the horizontal and vertical pixel location of the centre of the target, while the scale s and r represent the scale (area) and the aspect ratio of the target's bounding box respectively.
    Note that the aspect ratio is considered to be constant. When a detection is associated to a target, the detected bounding box is used to update the target state where the velocity components are solved optimally 
    via a Kalman filter framework. If no detection is associated to the target, its state is simply predicted without correction using the linear velocity model.
    """

    def __init__(self, max_age=3, min_hits=3, iou_threshold=0.3, logfile=None):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers:List[KalmanBoxTracker] = []
        self.frame_count = 0
        self.logger = TrackerLogger(logfile) if logfile else None


    def reset(self):
        self.trackers:List[KalmanBoxTracker] = []
        self.frame_count = 0


    def update(self, dets=np.empty((0, 6)), **kwargs):
        """
        Update the tracker state.

        Args:
            dets (np.ndarray): Detections with shape (N, 6) [[x1,y1,x2,y2,score,cls],[x1,y1,x2,y2,score,cls],...], default empty.

        Keyword Args:
            center (tuple): Optional, (x, y) rotation center.
            angle_deg (float): Optional, rotation in degrees.
            vx (float): Optional, velocity in x-direction.
            vy (float): Optional, velocity in y-direction.
        """
        self.frame_count += 1

        # Postprocess of the detections
        bboxes = dets[:, :4]        # x1,y1,x2,y2
        scores = dets[:, 4]         # score
        classes = dets[:, 5]        # class
        ids = range(dets.shape[0])  # auto-incremental id

        scores = np.expand_dims(scores, axis=-1)
        ids = np.expand_dims(ids, axis=-1)

        dets = np.column_stack((bboxes, scores, classes, ids))

        # Get predicted locations from existing trackers.
        trks_state = []
        to_del = []

        for i, tracker in enumerate(self.trackers):
            tracker.predict()

            tracker_state = tracker.obj()  # returns np.ndarray of shape (16,)
            trks_state.append(tracker_state)

            if np.any(np.isnan(tracker_state[:4])):  # Check bbox
                to_del.append(i)

        for idx in reversed(to_del):
            self.trackers.pop(idx)

        # Compute trks array (N, 5), where N is the number of trackers.
        trks_array = np.stack(trks_state, axis=0) if trks_state else np.empty((0, 16), dtype=np.float32)

        matched, unmatched_dets, unmatched_trks, iou_matrix = associate_detections_to_trackers(dets, trks_array, self.iou_threshold)

        if self.logger:
            extended_iou_matrix = TrackerLogger.build_extended_iou_matrix(dets, trks_array, iou_matrix)
            self.logger.save_logs(  
                self.frame_count,
                dets,
                trks_array,
                extended_iou_matrix,
            )
            self.logger.save_hungarian(matched, unmatched_dets, dets, trks_array)

        # Update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # Create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            x1, y1, x2, y2 = dets[i, :4]
            vx, vy = 0, 0
            p1 = (int((x2 + x1) / 2), int((y2 + y1) / 2))

            if 'center' in kwargs and 'angle_deg' in kwargs:
                p2 = self._rotate_point(center=kwargs['center'], point=p1, angle_deg=kwargs['angle_deg'])
                vx, vy = self._compute_velocity(p1, p2)

            elif 'vx' in kwargs and 'vy' in kwargs:
                vx, vy = kwargs['vx'], kwargs['vy']

            trk = KalmanBoxTracker(dets[i, :], vx, vy)
            self.trackers.append(trk)

        trks_array_valid = []
        for i in range(len(self.trackers) - 1, -1, -1):
            tracker_state = self.trackers[i].obj()

            # Remove dead tracklet
            if tracker_state[10] > self.max_age:
                self.trackers.pop(i)
                continue
            
            if tracker_state[9] > self.min_hits:
                trks_array_valid.append(tracker_state)

        if self.logger:
            self._compute_post_assignement_matrix(dets, trks_array_valid)

        return trks_array_valid, matched, unmatched_dets, unmatched_trks
    
    def _rotate_point(self, center, point, angle_deg):
        angle_rad = np.radians(angle_deg)
        
        # Translate point to origin
        translated_x = point[0] - center[0]
        translated_y = point[1] - center[1]
        
        # Apply rotation
        rotated_x = translated_x * np.cos(angle_rad) - translated_y * np.sin(angle_rad)
        rotated_y = translated_x * np.sin(angle_rad) + translated_y * np.cos(angle_rad)
        
        # Translate back
        final_x = rotated_x + center[0]
        final_y = rotated_y + center[1]
        
        return (final_x, final_y)
    
    def _compute_velocity(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        vx = (x2 - x1)
        vy = (y2 - y1)
        return vx, vy

    def _compute_post_assignement_matrix(self, dets, trks):
        if len(trks) == 0:
            iou_matrix = np.array([])
            self.logger._write_iou_matrix(
                iou_matrix,
                title="Post-Assignement-IOU Matrix  (After tracker.update() and initilization of unmatched detections)",
            )
            return

        iou_matrix = iou_batch(dets[:, 0:4], trks[:, 0:4])
        iou_matrix = TrackerLogger.build_extended_iou_matrix(dets, trks, iou_matrix)
        self.logger._write_iou_matrix(
            iou_matrix,
            title="Post-Assignement-IOU Matrix (After tracker.update() and initilization of unmatched detections)",
        )