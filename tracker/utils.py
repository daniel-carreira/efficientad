import numpy as np


def linear_assignment(cost_matrix):
    """
    Helper function for linear assignment optimization
    """
    try:
        import lap

        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])

    except ImportError:
        from scipy.optimize import linear_sum_assignment

        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)   
                                               
    return(o) 


def convert_bbox_to_z(bbox):
    """
    Converts a bounding box [x1, y1, x2, y2] to state vector [x, y, s, r] where:
    x, y = center of the box
    s = scale (area of the box)
    r = aspect ratio (width/height)
    """
    _bbox = np.array(bbox, dtype=np.float64)
    w = _bbox[2] - _bbox[0]
    h = _bbox[3] - _bbox[1]
    x = _bbox[0] + w / 2.0
    y = _bbox[1] + h / 2.0
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Converts a state vector [x, y, s, r] back to bounding box coordinates [x1, y1, x2, y2].
    If score is provided, the function returns [x1, y1, x2, y2, score].
    """
    w = np.sqrt(x[2] * x[3])  # Width of the box
    h = x[2] / w  # Height of the box

    if not score:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]
        ).reshape((1, 4))
    else:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]
        ).reshape((1, 5))
    
    
def associate_detections_to_trackers(dets, trks, iou_threshold=0.3):
    """
    Assigns detections to tracked objects (both as bounding boxes).

    Returns:
        matches:              Nx2 array of matched detection-tracker indices
        unmatched_detections: list of unmatched detection indices
        unmatched_trackers:   list of unmatched tracker indices
        iou_matrix:           MxN matrix of IoU values
    """
    if len(trks) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(dets)),
            np.empty((0, 5), dtype=int),
            np.array([]),
        )

    # Compute IoU matrix
    iou_matrix = iou_batch(dets[:, :4], trks[:, :4])

    # Determine initial matches
    if min(iou_matrix.shape) > 0:
        binary_iou = (iou_matrix > iou_threshold).astype(np.int32)
        if binary_iou.sum(1).max() == 1 and binary_iou.sum(0).max() == 1:
            matched_indices = np.stack(np.where(binary_iou), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty((0, 2), dtype=int)

    matched_dets = matched_indices[:, 0] if matched_indices.size > 0 else []
    matched_trks = matched_indices[:, 1] if matched_indices.size > 0 else []

    unmatched_dets = [d for d in range(len(dets)) if d not in matched_dets]
    unmatched_trks = [t for t in range(len(trks)) if t not in matched_trks]

    # Filter matches by IoU threshold
    matches = []
    for m in matched_indices:
        det_idx, trk_idx = m
        if iou_matrix[det_idx, trk_idx] < iou_threshold:
            unmatched_dets.append(det_idx)
            unmatched_trks.append(trk_idx)
        else:
            if dets[det_idx][5] == trks[trk_idx][5]:
                matches.append(m.reshape(1, 2))

    matches = np.concatenate(matches, axis=0) if matches else np.empty((0, 2), dtype=int)

    return (
        matches,
        np.array(unmatched_dets),
        np.array(unmatched_trks),
        iou_matrix,
    )