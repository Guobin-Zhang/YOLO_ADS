import torch
from torch.optim import AdamW
from yolov5_model import YOLOv5_GPU, YOLOv5_FPGA
from dataset_preprocessing import get_datasets, get_data_loaders
import torch.nn as nn

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification"""
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return loss.mean()

def train_model(use_fpga=False):
    """Train YOLOv5 model with strict hardware separation (FPGA/GPU)"""
    # Hardware mode check
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
    
    model = model.to(device)
    
    # Optimizer configuration
    optimizer = AdamW(model.parameters(), lr=1e-4 if use_fpga else 2e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    cls_criterion = FocalLoss()
    
    # Load datasets
    train_dataset, val_dataset, test_dataset = get_datasets()
    train_loader, val_loader, test_loader = get_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=32)
    
    # Training loop for 32x32 memristor array
    for epoch in range(300):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(data)
            loss = cls_criterion(outputs, target)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            # Frequent FPGA updates for small array
            if use_fpga and (batch_idx % 4 == 0):  # Update every 4 batches
                try:
                    model.update_fpga_weights()
                except RuntimeError as e:
                    print(f"FPGA weight update failed: {str(e)}")
                    return
            
            total_loss += loss.item()
        
        # Learning rate scheduling
        scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
    
    # Save GPU model
    if not use_fpga:
        torch.save(model.state_dict(), "yolov5_gpu.pth")
    else:
        print("FPGA training completed. Weights stored on hardware.")

if __name__ == "__main__":
    train_model(use_fpga=True)