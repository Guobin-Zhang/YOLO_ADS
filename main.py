import torch
import torch.nn as nn
import torchvision
from fpga_interface import FPGAInterface

class ChannelAttention(nn.Module):
    """Channel Attention Module from CBAM"""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """Spatial Attention Module from CBAM"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channels, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class GhostConv(nn.Module):
    """Ghost Convolution from GhostNet"""
    def __init__(self, in_ch, out_ch, kernel_size=1, ratio=2, dw_size=3):
        super().__init__()
        init_ch = out_ch // ratio
        new_ch = init_ch*(ratio-1)
        
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_ch, init_ch, kernel_size, stride=1, 
                     padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(init_ch),
            nn.ReLU(inplace=True)
        )
        
        self.cheap_conv = nn.Sequential(
            nn.Conv2d(init_ch, new_ch, dw_size, stride=1,
                     padding=dw_size//2, groups=init_ch, bias=False),
            nn.BatchNorm2d(new_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_conv(x1)
        return torch.cat([x1, x2], dim=1)

class DynamicHead(nn.Module):
    """Dynamic Head: Unifying Object Detection Heads with Attentions"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.cls_pred = nn.Conv2d(in_channels, num_classes, 1)
        self.reg_pred = nn.Conv2d(in_channels, 4, 1)

    def forward(self, x):
        scale_att = self.scale_attention(x)
        spatial_att = self.spatial_attention(x)
        cls_logits = self.cls_pred(x * scale_att)
        reg_pred = self.reg_pred(x * spatial_att)
        return torch.cat([reg_pred, cls_logits], dim=1)

class BiFPN(nn.Module):
    """Bidirectional Feature Pyramid Network"""
    def __init__(self, channels, levels=5):
        super().__init__()
        self.levels = levels
        self.conv6_up = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv5_up = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv4_up = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3_up = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv4_down = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv5_down = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv6_down = nn.Conv2d(channels, channels, 3, padding=1)
        self.epsilon = 1e-4

    def forward(self, inputs):
        p3, p4, p5, p6, p7 = inputs
        
        # Top-down path
        p7_up = nn.functional.interpolate(p7, scale_factor=2)
        p6_td = self.conv6_up(p6 + p7_up)
        p6_up = nn.functional.interpolate(p6_td, scale_factor=2)
        p5_td = self.conv5_up(p5 + p6_up)
        p5_up = nn.functional.interpolate(p5_td, scale_factor=2)
        p4_td = self.conv4_up(p4 + p5_up)
        p4_up = nn.functional.interpolate(p4_td, scale_factor=2)
        p3_out = self.conv3_up(p3 + p4_up)

        # Bottom-up path
        p3_dn = self.conv3_up(p3_out)
        p4_dn = self.conv4_down(p4_td + nn.functional.max_pool2d(p3_dn, 2))
        p5_dn = self.conv5_down(p5_td + nn.functional.max_pool2d(p4_dn, 2))
        p6_dn = self.conv6_down(p6_td + nn.functional.max_pool2d(p5_dn, 2))
        p7_out = self.conv6_down(p7 + nn.functional.max_pool2d(p6_dn, 2))

        return p3_out, p4_dn, p5_dn, p6_dn, p7_out

class YOLOv9(nn.Module):
    """Enhanced YOLOv9 model architecture with modern components"""
    def __init__(self, num_classes=5):
        super(YOLOv9, self).__init__()
        self.num_classes = num_classes
        
        # SRM-enhanced Data Preprocessing
        self.data_processing = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            CBAM(64),  # Added attention
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )
        
        # Backbone with GhostConv and CBAM
        self.backbone = nn.Sequential(
            self._make_ghost_layer(64, 128, 3),
            self._make_ghost_layer(128, 256, 4),
            self._make_ghost_layer(256, 512, 6),
            self._make_ghost_layer(512, 1024, 3)
        )
        
        # BiFPN Feature Fusion
        self.neck = BiFPN(1024)
        
        # Dynamic Detection Head
        self.head = DynamicHead(1024, num_classes)
        
        # Additional components
        self.dropout = nn.Dropout(0.2)
        self.aux_head = nn.Conv2d(1024, num_classes, 1)  # Auxiliary loss

    def _make_ghost_layer(self, in_ch, out_ch, blocks):
        layers = []
        layers.append(GhostConv(in_ch, out_ch))
        layers.append(nn.MaxPool2d(2))
        for _ in range(blocks):
            layers.append(GhostConv(out_ch, out_ch))
            layers.append(CBAM(out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.data_processing(x)
        c3 = self.backbone[0](x)
        c4 = self.backbone[1](c3)
        c5 = self.backbone[2](c4)
        c6 = self.backbone[3](c5)
        
        # Feature Pyramid
        features = self.neck([c3, c4, c5, c6, c6])  # Simplified input
        
        # Multi-scale predictions
        p3, p4, p5 = features[:3]
        main_out = self.head(p5)
        aux_out = self.aux_head(p3)
        return main_out, aux_out

class YOLOv9_GPU(YOLOv9):
    """GPU implementation with enhanced features"""
    def __init__(self, num_classes=5):
        super().__init__(num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == 'cuda')
        
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
            x = self.data_processing(x.to(self.device))
            return super().forward(x)

class YOLOv9_FPGA(YOLOv9):
    """FPGA implementation with analog SRM integration"""
    def __init__(self, num_classes=5):
        super().__init__(num_classes)
        self.fpga = FPGAInterface()
        if not self.fpga.connected:
            raise RuntimeError("FPGA not detected. Connect hardware to proceed.")
        self._configure_analog_components()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self._program_layers()
        
    def _configure_analog_components(self):
        """Configure FPGA for analog operations"""
        self.fpga.set_analog_mode(True)
        self.fpga.configure_noise_profile(0.1)
        
    def _program_layers(self):
        """Program FPGA crossbar with block-wise weight updates"""
        layer_weights = {
            'data_processing': self.data_processing[0].weight.data,
            'backbone': [layer[0].primary_conv[0].weight.data for layer in self.backbone],
            'neck': [self.neck.conv3_up.weight.data, 
                    self.neck.conv4_down.weight.data,
                    self.neck.conv5_down.weight.data],
            'head': self.head.cls_pred.weight.data
        }
        
        # Program each layer with block-wise updates
        for name, weights in layer_weights.items():
            if isinstance(weights, list):
                for idx, w in enumerate(weights):
                    try:
                        self.fpga.program_layer_weights(name, w)
                    except RuntimeError as e:
                        print(f"Layer {name} block {idx} programming failed: {str(e)}")
                        raise
            else:
                try:
                    self.fpga.program_layer_weights(name, weights)
                except RuntimeError as e:
                    print(f"Layer {name} programming failed: {str(e)}")
                    raise
                
    def update_fpga_weights(self):
        """Update FPGA weights during training iterations"""
        self._program_layers()
        
    def forward(self, x):
        if not self.fpga.supports_analog():
            raise RuntimeError("Analog SRM requires FPGA in analog mode")
        x = self.quant(x)
        x = super().forward(x)
        return self.dequant(x)