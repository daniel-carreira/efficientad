import torch
import numpy as np
from tqdm import tqdm
from enum import Enum

class ModelMode(Enum):
    TRAIN = "train"
    EVAL = "eval"

@torch.no_grad()
def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std, out_channels,
            q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
    
    teacher_output = teacher(image)
    student_output = student(image)
    autoencoder_output = autoencoder(image)

    teacher_output = (teacher_output - teacher_mean) / teacher_std

    map_st = torch.mean((teacher_output - student_output[:, :out_channels])**2,
                        dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output -
                         student_output[:, out_channels:])**2,
                        dim=1, keepdim=True)
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae

@torch.no_grad()
def map_normalization(data_loader, teacher, student, autoencoder,
                      teacher_mean, teacher_std, out_channels, device, desc='Map normalization'):
    maps_st = []
    maps_ae = []
    
    # Ignore augmented AE image
    for image, _ in tqdm(data_loader, desc=desc):
        image = image.to(device)
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, out_channels=out_channels)
        maps_st.append(map_st)
        maps_ae.append(map_ae)
    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)
    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)
    return q_st_start, q_st_end, q_ae_start, q_ae_end

@torch.no_grad()
def teacher_normalization(teacher, train_loader, device):
    mean_outputs = []
    
    for train_image, _ in tqdm(train_loader, desc='[Train] Computing mean of features'):
        train_image = train_image.to(device)
        teacher_output = teacher(train_image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for train_image, _ in tqdm(train_loader, desc='[Train] Computing std of features'):
        train_image = train_image.to(device)
        teacher_output = teacher(train_image)
        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance)
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std

def process_anomaly_map(anomaly_map, dims):
    (width, height) = dims
    anomaly_map = torch.nn.functional.pad(anomaly_map, (4, 4, 4, 4))
    anomaly_map = torch.nn.functional.interpolate(
        anomaly_map, (height, width), mode='bilinear')
    anomaly_map = anomaly_map[0, 0].cpu().numpy()
    anomaly_map = np.clip(anomaly_map, 0, None)

    return anomaly_map