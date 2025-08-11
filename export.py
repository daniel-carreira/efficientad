import os
import sys
import yaml
import torch
import pickle
import logging
import argparse
import torch_tensorrt
from typing import Any
from easydict import EasyDict
from torchvision import datasets
from modelopt.onnx.quantization.quantize import quantize

from networks import EfficientAD


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Export models to ONNX / TensorRT format."
    )
    parser.add_argument('--results', required=True, help="Directory where results are stored.")
    parser.add_argument('--device', default='cpu', help="['cpu', 'cuda:0', 'cuda:1', ...]")
    parser.add_argument('--type', type=str, choices=['pytorch-fp32', 'torchscript-fp32', 'onnx-fp32', 'tensorrt-fp32', 'tensorrt-fp16'], required=True, help="Export Model format and precision.")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose logging.")
    return parser.parse_args()


def load_and_transform_images(image_dir: str, transform: Any, calib_data_size: int = 100) -> Any:
    """
    Load and transform images from a directory using a specified transform.

    Args:
        image_dir (str): Directory containing images organized for ImageFolder.
        transform: Transformations to apply to the images.
        calib_data_size (int): Maximum number of images to load.

    Returns:
        numpy.ndarray: A batch of images as a numpy array.
    """
    dataset = datasets.ImageFolder(image_dir, transform=transform)
    images = [dataset[i][0] for i in range(min(calib_data_size, len(dataset)))]
    return torch.stack(images, dim=0).numpy()


def export_model_to_pytorch(model: torch.nn.Module, model_name: str, output_path: str) -> None:
    """
    Export to PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        model_name (str): Name of the model (used for the output file).
        output_path (str): Directory to save the exported PyTorch model.
    """
    os.makedirs(output_path, exist_ok=True)
    pytorch_path = os.path.join(output_path, f'{model_name}_pytorch_fp32.pt')
    torch.save(model, pytorch_path)
    
    logging.info(f"Model '{model_name}' exported to PyTorch at {pytorch_path}")

def export_model_to_torchscript(model: torch.nn.Module, model_name: str, dummy_input: torch.Tensor, output_path: str) -> None:
    """
    Export to PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        model_name (str): Name of the model (used for the output file).
        output_path (str): Directory to save the exported PyTorch model.
    """
    os.makedirs(output_path, exist_ok=True)
    ts_path = os.path.join(output_path, f'{model_name}_torchscript_fp32.pt')

    traced_script_module = torch.jit.trace(model, dummy_input)
    traced_script_module.save(ts_path)
    
    logging.info(f"Model '{model_name}' exported to TorchScript at {ts_path}")


def export_model_to_tensorrt(model: torch.nn.Module, model_name: str, dummy_input: torch.Tensor, output_path: str, precision_str: str) -> None:

    os.makedirs(output_path, exist_ok=True)
    tensorrt_path = os.path.join(output_path, f'{model_name}_tensorrt_{precision_str}.pt')

    precision = torch.float if precision_str == 'fp32' else torch.half 
    inputs = [dummy_input]

    trt_model = torch_tensorrt.compile(
        model,
        inputs=inputs,
        enabled_precisions={precision},
        truncate_long_and_double=True
    )
    torch_tensorrt.save(
        module=trt_model, 
        file_path=tensorrt_path, 
        output_format="torchscript", 
        inputs=inputs)

    logging.info(f"Model '{model_name}' exported to TensorRT at {tensorrt_path}")


def export_model_to_onnx(model: torch.nn.Module, model_name: str, dummy_input: torch.Tensor, output_path: str) -> None:
    """
    Export a PyTorch model to ONNX format.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        model_name (str): Name of the model (used for the output file).
        dummy_input (torch.Tensor): Dummy input tensor for tracing.
        output_path (str): Directory to save the exported ONNX model.
    """
    os.makedirs(output_path, exist_ok=True)
    onnx_path = os.path.join(output_path, f'{model_name}_onnx_fp32.onnx')
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamo=False
    )
    logging.info(f"Model '{model_name}' exported to ONNX at {onnx_path}")


def export_model_to_onnx_quant(args: argparse.Namespace, model_name: str, calib_path: str,
                               transform: Any, output_path: str) -> None:
    """
    Export and quantize a model to ONNX format.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
        model_name (str): Name of the model.
        calib_path (str): Path to calibration images.
        transform: Transformations to apply to calibration images.
        output_path (str): Directory to save the quantized ONNX model.
    """
    calib_data = load_and_transform_images(calib_path, transform, calib_data_size=100)
    onnx_path = os.path.join(output_path, f'{model_name}_weights_fp32_activations_fp32.onnx')
    quantization = args.type.split('-')[-1]
    quantized_onnx_path = os.path.join(output_path, f'{model_name}_weights_{quantization}_activations_fp32.onnx')

    calibration_methods = {
        'int8': 'entropy',
        'int4': 'awq_clip'
    }
    logging.info(f"Using device: cpu")

    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    quantize(
        onnx_path,
        quantize_mode=quantization,
        calibration_data=calib_data,
        calibration_method=calibration_methods.get(quantization),
        calibration_eps=['cpu'],
        output_path=quantized_onnx_path,
        nodes_to_exclude=[".*_output"],
        op_types_to_exclude=["Relu", "Clip", "Sigmoid", "Tanh"],
        verbose=args.verbose
    )

    sys.stdout = original_stdout


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load a PyTorch model from a file.

    Args:
        model_path (str): Path to the saved model file.
        device (torch.device): Device to load the model on.

    Returns:
        torch.nn.Module: Loaded PyTorch model in evaluation mode.
    """
    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        logging.info(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {e}")
        raise


def main():
    """
    Main function to export and optionally quantize PyTorch models to ONNX format.
    """
    args = parse_arguments()

    # Configure logging with the desired verbosity.
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    config_path = os.path.join(args.results, 'config.yml')
    try:
        with open(config_path, 'r') as f:
            config = EasyDict(yaml.safe_load(f))
        logging.info(f"Configuration loaded from {config_path}")
    except Exception as e:
        logging.error(f"Failed to load configuration from {config_path}: {e}")
        return

    input_path = os.path.join(args.results, 'train')
    output_path = os.path.join(args.results, 'export')

    device = torch.device(args.device)
    logging.info(f"Using device: {device}")

    # Load models to device
    try:
        models = {
            "teacher": load_model(os.path.join(input_path, 'teacher.pth'), device),
            "student": load_model(os.path.join(input_path, 'student_best.pth'), device),
            "autoencoder": load_model(os.path.join(input_path, 'autoencoder_best.pth'), device)
        }
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        return
    
    # Load configurations
    teacher_norm_path = os.path.join(input_path, 'teacher_normalization.pkl')
    map_norm_path = os.path.join(input_path, 'map_normalization_best.pkl')

    with open(teacher_norm_path, 'rb') as file:
        teacher_norm = pickle.load(file)

    with open(map_norm_path, 'rb') as file:
        map_norm = pickle.load(file)

    # Initialize model
    model = EfficientAD(models["teacher"], models["student"], models["autoencoder"], teacher_norm, map_norm)
    model = model.eval().to(device)

    # Create a dummy input for model export based on the configured input size
    input_size = (1, 3, config.net.input_size, config.net.input_size)
    dummy_input = torch.randn(*input_size).to(device)

    type_parts = args.type.split('-')
    export_type = type_parts[0]
    quantization = type_parts[-1]

    if export_type == 'tensorrt':
        export_model_to_tensorrt(
            model=model, 
            model_name="efficientad", 
            dummy_input=dummy_input, 
            output_path=output_path, 
            precision_str=quantization
        )
    
    elif export_type == 'onnx':
        export_model_to_onnx(
            model=model, 
            model_name="efficientad",
            dummy_input=dummy_input, 
            output_path=output_path
        )

    elif export_type == 'pytorch':
        export_model_to_pytorch(
            model=model,
            model_name="efficientad",
            output_path=output_path
        )

    elif export_type == 'torchscript':
        export_model_to_torchscript(
            model=model,
            model_name="efficientad",
            dummy_input=dummy_input,
            output_path=output_path
        )


if __name__ == '__main__':
    main()
