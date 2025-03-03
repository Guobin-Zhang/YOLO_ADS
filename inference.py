import torch
from yolov9_model import YOLOv9_GPU, YOLOv9_FPGA
from dataset_preprocessing import get_datasets, get_data_loaders

def inference(use_fpga=False):
    """Run inference with YOLOv9 model on FPGA or GPU"""
    device = torch.device("cpu" if use_fpga else "cuda")
    
    if use_fpga:
        try:
            model = YOLOv9_FPGA(num_classes=5)
        except RuntimeError as e:
            print(f"FPGA Error: {str(e)}")
            exit(1)
    else:
        model = YOLOv9_GPU(num_classes=5)
        model.load_state_dict(torch.load("yolov9_gpu.pth"))
    
    _, _, test_dataset = get_datasets()
    _, _, test_loader = get_data_loaders(None, None, test_dataset)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            predictions = output.argmax(dim=1)
            print(f"Predictions: {predictions.tolist()}")

if __name__ == "__main__":
    inference(use_fpga=True)