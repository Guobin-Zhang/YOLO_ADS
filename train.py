import torch
from torch.optim import AdamW, SGD
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from yolov9_model import YOLOv9_GPU, YOLOv9_FPGA
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
        loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return loss.mean()

def train_model(use_fpga=False):
    device = torch.device("cuda" if (torch.cuda.is_available() and not use_fpga) else "cpu")
    
    # Initialize model
    model = YOLOv9_GPU(num_classes=5) if not use_fpga else YOLOv9_FPGA(num_classes=5)
    model = model.to(device)
    
    # Add FPGA-specific training logic
    if use_fpga:
        optimizer = AdamW(model.parameters(), lr=1e-4)  # Lower learning rate for FPGA stability
        grad_accum_steps = 8  # More frequent FPGA updates
    else:
        optimizer = AdamW(model.parameters(), lr=2e-4)
        grad_accum_steps = 4
        
    # Training loop modifications
    for epoch in range(300):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                main_out, aux_out = model(data)
                loss = cls_criterion(main_out, target) + 0.3*cls_criterion(aux_out, target)
            
            # Backpropagation
            scaler.scale(loss).backward()
            
            # Gradient accumulation and FPGA updates
            if (batch_idx+1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Update FPGA weights iteratively
                if use_fpga:
                    try:
                        model.update_fpga_weights()
                    except RuntimeError as e:
                        print(f"Training aborted: {str(e)}")
                        return
                
                scheduler.step()