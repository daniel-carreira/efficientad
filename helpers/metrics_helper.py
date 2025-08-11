import numpy as np
from bisect import bisect
from scipy.ndimage.measurements import label
from sklearn.metrics import precision_recall_curve

def compute_classification_roc(
        anomaly_maps,
        scoring_function,
        ground_truth_labels):
    """Compute the ROC curve for anomaly classification on the image level.

    Args:
        anomaly_maps: List of anomaly maps (2D numpy arrays) that contain
          a real-valued anomaly score at each pixel.
        scoring_function: Function that turns anomaly maps into a single
          real valued anomaly score.

        ground_truth_labels: List of integers that indicate the ground truth
          class for each input image. 0 corresponds to an anomaly-free sample
          while a value != 0 indicates an anomalous sample.

    Returns:
        fprs: List of false positive rates.
        tprs: List of correspoding true positive rates.
    """
    assert len(anomaly_maps) == len(ground_truth_labels)

    # Compute the anomaly score for each anomaly map.
    anomaly_scores = map(scoring_function, anomaly_maps)
    num_scores = len(anomaly_maps)

    # Sort samples by anomaly score. Keep track of ground truth label.
    sorted_samples = \
        sorted(zip(anomaly_scores, ground_truth_labels), key=lambda x: x[0])

    # Compute the number of OK and NOK samples from the ground truth.
    ground_truth_labels_np = np.array(ground_truth_labels)
    num_nok = ground_truth_labels_np[ground_truth_labels_np != 0].size
    num_ok = ground_truth_labels_np[ground_truth_labels_np == 0].size

    # Initially, every NOK sample is correctly classified as anomalous
    # (tpr = 1.0), and every OK sample is incorrectly classified as anomalous
    # (fpr = 1.0).
    fprs = [1.0]
    tprs = [1.0]

    # Keep track of the current number of false and true positive predictions.
    num_fp = num_ok
    num_tp = num_nok

    # Compute new true and false positive rates when successively increasing
    # the threshold.
    next_score = None

    for i, (current_score, label) in enumerate(sorted_samples):

        if label == 0:
            num_fp -= 1
        else:
            num_tp -= 1

        if i < num_scores - 1:
            next_score = sorted_samples[i + 1][0]
        else:
            next_score = None  # end of list

        if (next_score != current_score) or (next_score is None):
            fprs.append(num_fp / num_ok)
            tprs.append(num_tp / num_nok)

    # Return (FPR, TPR) pairs in increasing order.
    fprs = fprs[::-1]
    tprs = tprs[::-1]

    return fprs, tprs

def compute_pro(anomaly_maps, ground_truth_maps):
    """Compute the PRO curve for a set of anomaly maps with corresponding ground
    truth maps.

    Args:
        anomaly_maps: List of anomaly maps (2D numpy arrays) that contain a
          real-valued anomaly score at each pixel.

        ground_truth_maps: List of ground truth maps (2D numpy arrays) that
          contain binary-valued ground truth labels for each pixel.
          0 indicates that a pixel is anomaly-free.
          1 indicates that a pixel contains an anomaly.

    Returns:
        fprs: numpy array of false positive rates.
        pros: numpy array of corresponding PRO values.
    """

    # Structuring element for computing connected components.
    structure = np.ones((3, 3), dtype=int)

    num_ok_pixels = 0
    num_gt_regions = 0

    shape = (len(anomaly_maps),
             anomaly_maps[0].shape[0],
             anomaly_maps[0].shape[1])
    fp_changes = np.zeros(shape, dtype=np.uint32)
    assert shape[0] * shape[1] * shape[2] < np.iinfo(fp_changes.dtype).max, \
        'Potential overflow when using np.cumsum(), consider using np.uint64.'

    pro_changes = np.zeros(shape, dtype=np.float64)

    for gt_ind, gt_map in enumerate(ground_truth_maps):

        # Compute the connected components in the ground truth map.
        labeled, n_components = label(gt_map, structure)
        num_gt_regions += n_components

        # Compute the mask that gives us all ok pixels.
        ok_mask = labeled == 0
        num_ok_pixels_in_map = np.sum(ok_mask)
        num_ok_pixels += num_ok_pixels_in_map

        # Compute by how much the FPR changes when each anomaly score is
        # added to the set of positives.
        # fp_change needs to be normalized later when we know the final value
        # of num_ok_pixels -> right now it is only the change in the number of
        # false positives
        fp_change = np.zeros_like(gt_map, dtype=fp_changes.dtype)
        fp_change[ok_mask] = 1

        # Compute by how much the PRO changes when each anomaly score is
        # added to the set of positives.
        # pro_change needs to be normalized later when we know the final value
        # of num_gt_regions.
        pro_change = np.zeros_like(gt_map, dtype=np.float64)
        for k in range(n_components):
            region_mask = labeled == (k + 1)
            region_size = np.sum(region_mask)
            pro_change[region_mask] = 1. / region_size

        fp_changes[gt_ind, :, :] = fp_change
        pro_changes[gt_ind, :, :] = pro_change

    # Flatten the numpy arrays before sorting.
    anomaly_scores_flat = np.array(anomaly_maps).ravel()
    fp_changes_flat = fp_changes.ravel()
    pro_changes_flat = pro_changes.ravel()

    # Sort all anomaly scores.
    sort_idxs = np.argsort(anomaly_scores_flat).astype(np.uint32)[::-1]

    # Info: np.take(a, ind, out=a) followed by b=a instead of
    # b=a[ind] showed to be more memory efficient.
    np.take(anomaly_scores_flat, sort_idxs, out=anomaly_scores_flat)
    anomaly_scores_sorted = anomaly_scores_flat
    np.take(fp_changes_flat, sort_idxs, out=fp_changes_flat)
    fp_changes_sorted = fp_changes_flat
    np.take(pro_changes_flat, sort_idxs, out=pro_changes_flat)
    pro_changes_sorted = pro_changes_flat

    del sort_idxs

    # Get the (FPR, PRO) curve values.
    np.cumsum(fp_changes_sorted, out=fp_changes_sorted)
    fp_changes_sorted = fp_changes_sorted.astype(np.float32, copy=False)
    np.divide(fp_changes_sorted, num_ok_pixels, out=fp_changes_sorted)
    fprs = fp_changes_sorted

    np.cumsum(pro_changes_sorted, out=pro_changes_sorted)
    np.divide(pro_changes_sorted, num_gt_regions, out=pro_changes_sorted)
    pros = pro_changes_sorted

    # Merge (FPR, PRO) points that occur together at the same threshold.
    # For those points, only the final (FPR, PRO) point should be kept.
    # That is because that point is the one that takes all changes
    # to the FPR and the PRO at the respective threshold into account.
    # -> keep_mask is True if the subsequent score is different from the
    # score at the respective position.
    # anomaly_scores_sorted = [7, 4, 4, 4, 3, 1, 1]
    # ->          keep_mask = [T, F, F, T, T, F]
    keep_mask = np.append(np.diff(anomaly_scores_sorted) != 0, np.True_)
    del anomaly_scores_sorted

    fprs = fprs[keep_mask]
    pros = pros[keep_mask]
    del keep_mask

    # To mitigate the adding up of numerical errors during the np.cumsum calls,
    # make sure that the curve ends at (1, 1) and does not contain values > 1.
    np.clip(fprs, a_min=None, a_max=1., out=fprs)
    np.clip(pros, a_min=None, a_max=1., out=pros)

    # Make the fprs and pros start at 0 and end at 1.
    zero = np.array([0.])
    one = np.array([1.])

    return np.concatenate((zero, fprs, one)), np.concatenate((zero, pros, one))

def trapezoid(x, y, x_max=None):
    """
    This function calculates the definit integral of a curve given by
    x- and corresponding y-values. In contrast to, e.g., 'numpy.trapz()',
    this function allows to define an upper bound to the integration range by
    setting a value x_max.

    Points that do not have a finite x or y value will be ignored with a
    warning.

    Args:
        x: Samples from the domain of the function to integrate
          Need to be sorted in ascending order. May contain the same value
          multiple times. In that case, the order of the corresponding
          y values will affect the integration with the trapezoidal rule.
        y: Values of the function corresponding to x values.
        x_max: Upper limit of the integration. The y value at max_x will be
          determined by interpolating between its neighbors. Must not lie
          outside of the range of x.

    Returns:
        Area under the curve.
    """

    x = np.asarray(x)
    y = np.asarray(y)
    finite_mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    if not finite_mask.all():
        print("WARNING: Not all x and y values passed to trapezoid(...)"
              " are finite. Will continue with only the finite values.")
    x = x[finite_mask]
    y = y[finite_mask]

    # Introduce a correction term if max_x is not an element of x.
    correction = 0.
    if x_max is not None:
        if x_max not in x:
            # Get the insertion index that would keep x sorted after
            # np.insert(x, ins, x_max).
            ins = bisect(x, x_max)
            # x_max must be between the minimum and the maximum, so the
            # insertion_point cannot be zero or len(x).
            assert 0 < ins < len(x)

            # Calculate the correction term which is the integral between
            # the last x[ins-1] and x_max. Since we do not know the exact value
            # of y at x_max, we interpolate between y[ins] and y[ins-1].
            y_interp = y[ins - 1] + ((y[ins] - y[ins - 1]) *
                                     (x_max - x[ins - 1]) /
                                     (x[ins] - x[ins - 1]))
            correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])

        # Cut off at x_max.
        mask = x <= x_max
        x = x[mask]
        y = y[mask]

    # Return area under the curve using the trapezoidal rule.
    return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction

def compute_f1_optimal(anomaly_maps, scoring_function, ground_truth_labels):
    """
    Compute the optimal F1 score and the corresponding threshold.

    Args:
        anomaly_maps (np.ndarray): Predicted anomaly scores or probabilities.
        ground_truth_maps (np.ndarray): Ground truth binary labels.

    Returns:
        tuple[float, float]: A tuple containing the maximum F1 score and the threshold that achieves this score.
    """
    # Compute precision-recall curve
    anomaly_scores = list(map(scoring_function, anomaly_maps))
    precision, recall, thresholds = precision_recall_curve(ground_truth_labels, anomaly_scores)

    # Compute F1 scores for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

    # Find max F1 score
    max_f1_score = np.max(f1_scores)

    # Get indices of all occurrences of max F1
    max_f1_indices = np.where(f1_scores == max_f1_score)[0]

    # Find the index corresponding to the lowest threshold
    min_threshold_index = max_f1_indices[np.argmin(thresholds[max_f1_indices])]

    # Get the optimal threshold
    optimal_threshold = thresholds[min_threshold_index]

    return max_f1_score, optimal_threshold