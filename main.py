import torch
import torch.nn as nn
from torchvision.ops import Conv2dNormActivation
from fpga_interface import FPGAInterface

class Focus(nn.Module):
    """Focus module from YOLOv5 for spatial reduction"""
    def __init__(self, in_ch, out_ch, kernel_size=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch * 4, out_ch, kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        # Shape transform: (b,c,w,h) -> (b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat([patch_top_left, patch_top_right, patch_bot_left, patch_bot_right], dim=1)
        return self.conv(x)

class C3(nn.Module):
    """C3 module from YOLOv5 with 3 convolutions"""
    def __init__(self, in_ch, out_ch, n=1, shortcut=True):
        super().__init__()
        hidden_ch = out_ch // 2
        self.cv1 = Conv2dNormActivation(in_ch, hidden_ch, 1)
        self.cv2 = Conv2dNormActivation(in_ch, hidden_ch, 1)
        self.cv3 = Conv2dNormActivation(2 * hidden_ch, out_ch, 1)
        self.m = nn.Sequential(*[Conv2dNormActivation(hidden_ch, hidden_ch, 3) for _ in range(n)])

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.m(self.cv2(x))
        return self.cv3(torch.cat((x1, x2), dim=1))

class SPPF(nn.Module):
    """SPPF module from YOLOv5 for spatial pyramid pooling"""
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        hidden_ch = in_ch // 2
        self.cv1 = Conv2dNormActivation(in_ch, hidden_ch, 1)
        self.cv2 = Conv2dNormActivation(hidden_ch * 4, out_ch, 1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))

class YOLOv5(nn.Module):
    """YOLOv5 model architecture with FPGA-compatible layers"""
    def __init__(self, num_classes=5):
        super().__init__()
        # Data preprocessing layer
        self.data_processing = nn.Sequential(
            Focus(3, 64),
            Conv2dNormActivation(64, 128, 3, stride=2),
            C3(128, 128, n=3)
        )
        
        # Backbone
        self.backbone = nn.Sequential(
            Conv2dNormActivation(128, 256, 3, stride=2),
            C3(256, 256, n=6),
            Conv2dNormActivation(256, 512, 3, stride=2),
            C3(512, 512, n=9),
            Conv2dNormActivation(512, 1024, 3, stride=2),
            C3(1024, 1024, n=3),
            SPPF(1024, 1024)
        )
        
        # Neck (FPN)
        self.neck = nn.Sequential(
            Conv2dNormActivation(1024, 512, 1),
            nn.Upsample(scale_factor=2),
            C3(1024, 512, shortcut=False),
            Conv2dNormActivation(512, 256, 1),
            nn.Upsample(scale_factor=2),
            C3(512, 256, shortcut=False)
        )
        
        # Head (Detection and Classification)
        self.head = nn.Sequential(
            Conv2dNormActivation(256, 256, 3),
            nn.Conv2d(256, (5 + num_classes) * 3, 1)  # 3 anchors per scale
        )
        
    def forward(self, x):
        x = self.data_processing(x)
        x = self.backbone(x)
        x = self.neck(x)
        return self.head(x)

class YOLOv5_GPU(YOLOv5):
    """GPU implementation of YOLOv5 without FPGA integration"""
    def __init__(self, num_classes=5):
        super().__init__(num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def forward(self, x):
        return super().forward(x.to(self.device))

class YOLOv5_FPGA(YOLOv5):
    """FPGA implementation with memristor crossbar integration"""
    def __init__(self, num_classes=5):
        super().__init__(num_classes)
        self.fpga = FPGAInterface()
        if not self.fpga.connected:
            raise RuntimeError("FPGA not detected. Connect hardware to proceed.")
        self._configure_analog_components()
        self._program_layers()
        
    def _configure_analog_components(self):
        """Configure FPGA for analog operations"""
        self.fpga.set_analog_mode(True)
        self.fpga.configure_noise_profile(0.1)
        
    def _program_layers(self):
        """Program FPGA crossbar for four key layers: data_processing, backbone, neck, head"""
        layer_weights = {
            'data_processing': [
                self.data_processing[0].conv.weight.data,
                self.data_processing[1].conv.weight.data
            ],
            'backbone': [layer.conv.weight.data for layer in self.backbone if isinstance(layer, Conv2dNormActivation)],
            'neck': [layer.conv.weight.data for layer in self.neck if isinstance(layer, Conv2dNormActivation)],
            'head': self.head[0].conv.weight.data
        }
        
        # Program each layer with 32x32 block-wise updates
        for name, weights in layer_weights.items():
            if isinstance(weights, list):
                for idx, w in enumerate(weights):
                    self.fpga.program_layer_weights(name, w)
            else:
                self.fpga.program_layer_weights(name, weights)
                
    def update_fpga_weights(self):
        """Update FPGA weights during training iterations"""
        self._program_layers()