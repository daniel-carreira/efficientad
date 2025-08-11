import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from .common import predict, map_normalization, process_anomaly_map
from helpers.metrics_helper import compute_classification_roc, compute_pro, compute_f1_optimal, trapezoid
from helpers.annotation_helper import load_yolo_annotation_to_matrix

class Evaluator:
    def __init__(self, config, test_set, default_transform, device, output_dir):
        self.config = config
        self.test_set = test_set
        self.default_transform = default_transform
        self.device = device
        self.output_dir = output_dir
        self.integration_limit = 1

        self.result_str = ""

        self._setup_log_file()

    def eval(self, efficientad=None, teacher=None, student=None, autoencoder=None, map_norm_loader=None, save_imgs=False, desc='[Test]'):
        roi_width, roi_height, _, _ = self._get_roi_dims()

        self.desc = desc

        self.per_class_scores_all = {}
        self.per_class_y_true_label = {}
        self.per_class_y_true_mask = {}
        self.per_class_scores_with_mask = {}

        if map_norm_loader is not None:
            self._compute_map_norm(data_loader=map_norm_loader, teacher=teacher, student=student, autoencoder=autoencoder, desc=desc)

        for image, target, path in tqdm(self.test_set, desc=f'{self.desc} Inference'):
            image_tensor = self.default_transform(image)
            image_tensor = image_tensor[None]
            image_tensor = image_tensor.to(self.device)

            if efficientad is not None:
                map_combined = efficientad(image_tensor)
                map_combined = torch.nn.functional.interpolate(map_combined, (roi_height, roi_width), mode='bilinear', align_corners=False)
                map_combined = map_combined[0, 0].detach().cpu().numpy()

            else:
                map_combined, map_st, map_ae = predict(
                    image=image_tensor, teacher=teacher, student=student,
                    autoencoder=autoencoder, teacher_mean=self.teacher_mean,
                    teacher_std=self.teacher_std, out_channels=self.config.net.out_channels, q_st_start=self.q_st_start, q_st_end=self.q_st_end,
                    q_ae_start=self.q_ae_start, q_ae_end=self.q_ae_end)
            
                map_combined = process_anomaly_map(map_combined, dims=(roi_width, roi_height))

            map_combined = map_combined.astype(np.float16)

            # Image label
            class_name = os.path.basename(os.path.dirname(path))
            y_true_label = 0 if class_name == 'normal' else 1

            # Load annotations if any
            annotation_path = os.path.splitext(path)[0] + ".txt"
            y_true_mask, has_defect = load_yolo_annotation_to_matrix(annotation_path=annotation_path, annotation_dims=image.size, roi_dims=self._get_roi_dims(), class_name=class_name)
            has_annotation = has_defect != None

            # Ensure all classes are initialized
            if class_name not in self.per_class_y_true_mask:
                self.per_class_scores_all[class_name] = []
                self.per_class_scores_with_mask[class_name] = []
                self.per_class_y_true_mask[class_name] = []
                self.per_class_y_true_label[class_name] = []

            # Append values
            self.per_class_scores_all[class_name].append(map_combined)
            self.per_class_y_true_label[class_name].append(y_true_label)
            if has_annotation:
                self.per_class_y_true_mask[class_name].append(y_true_mask)
                self.per_class_scores_with_mask[class_name].append(map_combined)

            if save_imgs:
                # Re-scale value for visualization porpuses
                map_combined_trim = np.where(map_combined > 1, 1, map_combined)
                map_scaled = (map_combined_trim * 255).astype(np.uint8)
                self._save_images(image_path=path, class_name=class_name, anomaly_map=map_scaled, gt_mask=y_true_mask)

        msg = "================================================================================="
        self._write_to_log(msg, with_print=True)
        self.result_str = msg + "\n"

        has_any_annotation = False

        # Compute per class metrics
        for class_name in self.per_class_y_true_label.keys():
            if class_name == 'normal':
                continue

            # ROC
            anomaly_maps = self.per_class_scores_all['normal'] + self.per_class_scores_all[class_name]
            gt_labels = self.per_class_y_true_label['normal'] + self.per_class_y_true_label[class_name]
            all_fprs, all_tprs = compute_classification_roc(anomaly_maps=anomaly_maps, scoring_function=np.max, ground_truth_labels=gt_labels)

            # AU-ROC
            au_roc = trapezoid(all_fprs, all_tprs) * 100
            msg = f'[{class_name}] AU-ROC: {au_roc:.4f}%'
            self._write_to_log(msg, with_print=True)
            self.result_str += msg + "\n"

            # F1
            f1_score, threshold = compute_f1_optimal(anomaly_maps=anomaly_maps, scoring_function=np.max, ground_truth_labels=gt_labels)
            msg = f'[{class_name}] F1-Score (threshold={threshold:.2f}): {f1_score*100:.4f}%'
            self._write_to_log(msg, with_print=True)
            self.result_str += msg + "\n"

            n_segmentations = len(self.per_class_scores_with_mask[class_name])
            if n_segmentations > 0:
                # PRO
                anomaly_maps = self.per_class_scores_with_mask[class_name]
                gt_masks = self.per_class_y_true_mask[class_name]
                all_fprs, all_pros = compute_pro(anomaly_maps=anomaly_maps, ground_truth_maps=gt_masks)

                # AU-PRO
                au_pro = trapezoid(all_fprs, all_pros, x_max=self.integration_limit) * 100
                au_pro /= self.integration_limit
                msg = f'[{class_name}] AU-PRO: {au_pro:.4f}%'
                self._write_to_log(msg, with_print=True)
                self.result_str += msg + "\n"

                has_any_annotation = True

        msg = "---------------------------------------------------------------------------------"
        self._write_to_log(msg, with_print=True)
        self.result_str += msg + "\n"

        # Compute overall metrics
        anomaly_maps = np.concatenate(list(self.per_class_scores_all.values()))
        gt_labels = np.concatenate(list(self.per_class_y_true_label.values()))

        # - ROC
        roc_curve = compute_classification_roc(anomaly_maps=anomaly_maps, scoring_function=np.max, ground_truth_labels=gt_labels)

        # AU-ROC
        au_roc = trapezoid(roc_curve[0], roc_curve[1]) * 100
        msg = f'[Overall] AU-ROC: {au_roc:.4f}%'
        self._write_to_log(msg, with_print=True)
        self.result_str += msg + "\n"

        # F1
        f1_score, threshold = compute_f1_optimal(anomaly_maps=anomaly_maps, scoring_function=np.max, ground_truth_labels=gt_labels)
        msg = f'[Overall] F1-Score (threshold={threshold:.2f}): {f1_score*100:.4f}%'
        self._write_to_log(msg, with_print=True)
        self.result_str += msg + "\n"

        # - PRO
        au_pro = None
        pro_curve = None
        if has_any_annotation:
            anomaly_maps = np.concatenate(list(self.per_class_scores_with_mask.values()))
            gt_masks = np.concatenate(list(self.per_class_y_true_mask.values()))
            pro_curve = compute_pro(anomaly_maps=anomaly_maps, ground_truth_maps=gt_masks)

            # AU-PRO
            au_pro = trapezoid(pro_curve[0], pro_curve[1], x_max=self.integration_limit) * 100
            au_pro /= self.integration_limit
            msg = f'[Overall] AU-PRO: {au_pro:.4f}%'
            self._write_to_log(msg, with_print=True)
            self.result_str += msg + "\n"

        msg = "=================================================================================\n"
        self._write_to_log(msg, with_print=True)
        self.result_str += msg

        # Save the graph corresponding to each curve
        self._draw_metrics(au_pro, pro_curve, au_roc, roc_curve)
        
        return au_pro, pro_curve, au_roc, roc_curve
    
    def to_str(self):
        return self.result_str
    
    def compute_actual_target_maps(self, teacher, student, autoencoder):
        roi_width, roi_height, _, _ = self._get_roi_dims()

        actual_maps = []
        target_maps = []
        image_paths = []

        for image, target, path in tqdm(self.test_set, desc=f'Inference'):
            image_tensor = self.default_transform(image)
            image_tensor = image_tensor[None]
            image_tensor = image_tensor.to(self.device)

            _, map_st, map_ae = predict(
                image=image_tensor, teacher=teacher, student=student,
                autoencoder=autoencoder, teacher_mean=self.teacher_mean,
                teacher_std=self.teacher_std, out_channels=self.config.net.out_channels, q_st_start=self.q_st_start, q_st_end=self.q_st_end,
                q_ae_start=self.q_ae_start, q_ae_end=self.q_ae_end)
        
            map_st = process_anomaly_map(map_st, dims=(roi_width, roi_height))
            map_ae = process_anomaly_map(map_ae, dims=(roi_width, roi_height))
            actual_maps.append((map_st, map_ae))

            class_name = os.path.basename(os.path.dirname(path))

            annotation_path = os.path.splitext(path)[0] + ".txt"
            y_true_mask, has_defect = load_yolo_annotation_to_matrix(annotation_path=annotation_path, annotation_dims=image.size, roi_dims=self._get_roi_dims(), class_name=class_name)
            y_true_mask = np.ones_like(y_true_mask) if has_defect == None and class_name != 'normal' else y_true_mask

            target_maps.append(y_true_mask)
            image_paths.append(path)

        return actual_maps, target_maps, image_paths

    def set_teacher_norm(self, teacher_mean, teacher_std):
        self.teacher_mean = teacher_mean
        self.teacher_std = teacher_std

    def set_map_norm(self, q_st_start, q_st_end, q_ae_start, q_ae_end):
        self.q_st_start = q_st_start
        self.q_st_end = q_st_end
        self.q_ae_start = q_ae_start
        self.q_ae_end = q_ae_end

    def load_teacher_norm_from_file(self, path):
        with open(path, 'rb') as file:
            teacher_mean, teacher_std = pickle.load(file)
            self.set_teacher_norm(teacher_mean.to(self.device), teacher_std.to(self.device))
        
        return teacher_mean, teacher_std

    def load_map_norm_from_file(self, path):
        with open(path, 'rb') as file:
            q_st_start, q_st_end, q_ae_start, q_ae_end = pickle.load(file)
            self.set_map_norm(q_st_start.to(self.device), q_st_end.to(self.device), q_ae_start.to(self.device), q_ae_end.to(self.device))
        
        return q_st_start, q_st_end, q_ae_start, q_ae_end

    def _compute_map_norm(self, data_loader, teacher, student, autoencoder, desc='[Test]'):
        q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
            data_loader=data_loader, teacher=teacher,
            student=student, autoencoder=autoencoder,
            teacher_mean=self.teacher_mean, teacher_std=self.teacher_std, out_channels=self.config.net.out_channels, device=self.device,
            desc=f'{desc} Map normalization')
        self.set_map_norm(q_st_start, q_st_end, q_ae_start, q_ae_end)
        
        return q_st_start, q_st_end, q_ae_start, q_ae_end
    
    def save_map_norm(self, path):
        with open(path, 'wb') as file:
            pickle.dump((self.q_st_start, self.q_st_end, self.q_ae_start, self.q_ae_end), file)

    def _setup_log_file(self):
        self.log_file = os.path.join(self.output_dir, 'log.txt')

    def _write_to_log(self, message, with_print=False):
        with open(self.log_file, mode="a") as f:
            f.write(f'{message}\n')
            if with_print:
                print(message)

    def _get_roi_dims(self):
        roi_values = self.config.dataset.roi.split(":")
        return map(int, roi_values)

    def _save_images(self, image_path, class_name, anomaly_map, gt_mask):
        roi_width, roi_height, roi_x_offset, roi_y_offset = self._get_roi_dims()

        image_name = os.path.basename(image_path)
        output_dir_test_dir = os.path.join(self.output_dir, class_name)
        os.makedirs(output_dir_test_dir, exist_ok=True)

        # Load the original image
        original_image = cv2.imread(image_path)

        # Convert anomaly map to 3-channel grayscale
        anomaly_map_grayscale = cv2.cvtColor(anomaly_map, cv2.COLOR_GRAY2BGR)

        mask_scaled = (gt_mask * 255).astype(np.uint8)

        # Create a colored heatmap using the mask
        colored_mask = np.zeros(anomaly_map_grayscale.shape, dtype=np.uint8)
        colored_mask[:,:,2] = mask_scaled

        # Overlay the heatmap on the original image
        anomaly_map_grayscale_with_gt = cv2.addWeighted(anomaly_map_grayscale, 1, colored_mask, 0.3, 0)

        # Create a black canvas (all zeros) with the same size as the original image
        anomaly_map_grayscale_full = np.zeros_like(original_image)
        anomaly_map_grayscale_full[roi_y_offset:roi_y_offset + roi_height, roi_x_offset:roi_x_offset + roi_width] = anomaly_map_grayscale_with_gt

        # Concatenate images side by side
        combined_image = np.hstack((original_image, anomaly_map_grayscale_full))

        # Save the combined image
        cv2.imwrite(os.path.join(output_dir_test_dir, image_name), combined_image)

    def _draw_metrics(self, au_pro, pro_curve, au_roc, roc_curve):

        if au_pro:
            # Plot PRO Curve
            plt.figure(figsize=(8, 6))
            plt.plot(pro_curve[0], pro_curve[1], label=f'PRO Curve (AUC = {au_pro:.2f})', color='blue')
            plt.xlim(0 - self.integration_limit * 0.02, self.integration_limit + self.integration_limit * 0.02)
            plt.ylim(-0.02, 1.02)
            plt.xlabel('Global FPR')
            plt.ylabel('Averaged Per-Region TPR')
            plt.title('PRO Curve')
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, 'pro_curve.jpg'), dpi=300)
            plt.close()

        # Plot ROC Curve
        plt.figure(figsize=(8, 6))
        plt.plot(roc_curve[0], roc_curve[1], label=f'ROC Curve (AUC = {au_roc:.2f})', color='red')
        plt.xlim(-0.02, 1.02)
        plt.ylim(-0.02, 1.02)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'roc_curve.jpg'), dpi=300)
        plt.close()