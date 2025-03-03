import torch
from yolov5_model import YOLOv5_GPU, YOLOv5_FPGA
from dataset_preprocessing import get_datasets, get_data_loaders

def inference(use_fpga=False):
    """Run inference with strict hardware separation (FPGA/CPU or GPU)"""
    if use_fpga and torch.cuda.is_available():
        raise RuntimeError("FPGA mode cannot coexist with GPU. Disable CUDA.")
    
    device = torch.device("cpu" if use_fpga else "cuda")
    
    # Initialize model
    if use_fpga:
        try:
            model = YOLOv5_FPGA(num_classes=5)
        except RuntimeError as e:
            print(f"FPGA Error: {str(e)}")
            exit(1)
    else:
        model = YOLOv5_GPU(num_classes=5)
        model.load_state_dict(torch.load("yolov5_gpu.pth"))
    
    # Load test data
    _, _, test_dataset = get_datasets()
    _, _, test_loader = get_data_loaders(None, None, test_dataset)

    # Inference loop
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data)
            predictions = output.argmax(dim=1)
            print(f"Predictions: {predictions.tolist()}")

if __name__ == "__main__":
    inference(use_fpga=True)