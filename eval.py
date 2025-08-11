import os
import yaml
import torch
import argparse
from easydict import EasyDict

from modules.trainer import Evaluator
from data.dataloader import ImageFolderWithPath, get_default_transforms


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--results', required=True)
    return parser.parse_args()

def main():
    # Load configuration
    args = get_argparse()

    # Open config file
    with open(os.path.join(args.results, 'config.yml')) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract ROI dims
    roi_values = config.dataset.roi.split(":")
    width, height, x_offset, y_offset = map(int, roi_values)

    # Load dataset
    default_transforms = get_default_transforms(config=config, roi_dims=(width, height, x_offset, y_offset))

    dataset_test = os.path.join(args.dataset, 'test')
    if not os.path.exists(dataset_test):
        dataset_test = args.dataset
    test_set = ImageFolderWithPath(dataset_test)

    # Default paths
    input_dir_train = os.path.join(args.results, 'train')
    output_dir_test = os.path.join(args.results, 'test')
    os.makedirs(output_dir_test, exist_ok=True)

    # ================ FLOAT 32 (PyTorch) ================
    # Load models
    teacher = torch.load(os.path.join(input_dir_train, 'teacher.pth'), map_location="cpu", weights_only=False).to(device).eval()
    student = torch.load(os.path.join(input_dir_train, 'student_best.pth'), map_location="cpu", weights_only=False).to(device).eval()
    autoencoder = torch.load(os.path.join(input_dir_train, 'autoencoder_best.pth'), map_location="cpu", weights_only=False).to(device).eval()

    # Init Evaluator
    evaluator = Evaluator(config=config, test_set=test_set, default_transform=default_transforms, device=device, output_dir=output_dir_test)

    # Load teacher norm and map norm from path
    evaluator.load_teacher_norm_from_file(os.path.join(input_dir_train, 'teacher_normalization.pkl'))
    evaluator.load_map_norm_from_file(os.path.join(input_dir_train, 'map_normalization_best.pkl'))

    # Evaluate
    evaluator.eval(teacher=teacher, student=student, autoencoder=autoencoder, save_imgs=True)

    

if __name__ == '__main__':
    main()