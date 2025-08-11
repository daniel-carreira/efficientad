import torch
from torchinfo import summary

from .pdn import PDN
from .ae import AE


def initialize_models(config, with_print=False):
    """Initializes and returns the teacher, student, and autoencoder models."""
    teacher = PDN(size=config.net.type, out_channels=config.net.out_channels)
    student = PDN(size=config.net.type, out_channels=2*config.net.out_channels)
    
    state_dict = torch.load(config.net.teacher_weights, map_location="cpu", weights_only=True)
    teacher.load_state_dict(state_dict)

    autoencoder = AE(out_channels=config.net.out_channels)

    if with_print:
        input_shape = (1, 3, config.net.input_size, config.net.input_size)
        
        print("################################=- Auto-Encoder -=################################")
        summary(autoencoder, input_size=input_shape)

        print("\n################################=- Teacher -=################################")
        summary(teacher, input_size=input_shape)

        print("\n################################=- Student -=################################")
        summary(student, input_size=input_shape)
        print("")

    return teacher, student, autoencoder