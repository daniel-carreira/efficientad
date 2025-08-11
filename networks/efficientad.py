import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientAD(nn.Module):
    def __init__(self, teacher, student, autoencoder, teacher_norm, map_norm):
        super().__init__()

        self.teacher = teacher
        self.student = student
        self.autoencoder = autoencoder

        # Register mean and std as buffers so they move with self.to(device)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Register teacher normalization statistics
        self.register_buffer('teacher_mean', teacher_norm[0])
        self.register_buffer('teacher_std', teacher_norm[1])

        # Register normalization values for anomaly maps
        self.register_buffer('q_st_start', map_norm[0])
        self.register_buffer('q_st_end', map_norm[1])
        self.register_buffer('q_ae_start', map_norm[2])
        self.register_buffer('q_ae_end', map_norm[3])

    def forward(self, image_tensor):

        # =============== PRE-PROCESS ===============

        image_tensor_resized_norm = (image_tensor - self.mean) / self.std


        # =============== INFERENCE ===============

        teacher_output = self.teacher(image_tensor_resized_norm)
        student_output = self.student(image_tensor_resized_norm)
        autoencoder_output = self.autoencoder(image_tensor_resized_norm)

        teacher_output = (teacher_output - self.teacher_mean) / self.teacher_std

        map_st = torch.mean((teacher_output - student_output[:, :384])**2,
                            dim=1, keepdim=True)
        map_ae = torch.mean((autoencoder_output - student_output[:, 384:])**2,
                            dim=1, keepdim=True)
        
        map_st = 0.1 * (map_st - self.q_st_start) / (self.q_st_end - self.q_st_start)
        map_ae = 0.1 * (map_ae - self.q_ae_start) / (self.q_ae_end - self.q_ae_start)

        map_combined = 0.5 * map_st + 0.5 * map_ae


        # =============== POST-PROCESS ===============

        map_combined = F.pad(map_combined, (4, 4, 4, 4))
        #map_combined = torch.clamp(map_combined, min=0)

        #orig_height, orig_width = image_tensor.shape[-2], image_tensor.shape[-1]
        #map_combined = F.interpolate(map_combined, size=(orig_height, orig_width), mode="bilinear")
        
        return map_combined