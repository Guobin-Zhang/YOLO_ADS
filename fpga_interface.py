import torch
import numpy as np
import usb.core
import time

class FPGAInterface:
    """Hardware interface for 32x32 memristor crossbar array"""
    def __init__(self):
        self.connected = self._initialize_hardware()
        self.analog_mode_enabled = False
        self.layer_mapping = {
            'data_processing': 0x01,
            'backbone': 0x02,
            'neck': 0x03,
            'head': 0x04,
            'classifier': 0x05
        }
        
    def _initialize_hardware(self):
        """Initialize connection to FPGA with exact hardware specifications"""
        try:
            # FTDI FT601 USB3.0 Controller
            self.dev = usb.core.find(idVendor=0x0403, idProduct=0x6010)
            if self.dev is None:
                return False
            
            # Send hardware handshake
            self._send_control_packet(0xA0, 0x01, 0x00, b'INIT', timeout=5000)
            
            # Configure crossbar parameters
            self._configure_dac_settings()
            return True
        except Exception as e:
            print(f"FPGA initialization failed: {str(e)}")
            return False
            
    def _configure_dac_settings(self):
        """Configure DAC parameters per hardware specs"""
        # AD5764R DAC configuration (±10V range)
        self._send_control_packet(0xB0, 0x00, 0x00, b'\x01\x00')  # Channel 1: +2V pulse
        self._send_control_packet(0xB0, 0x00, 0x01, b'\x02\x00')  # Channel 2: -2V pulse
        self._send_control_packet(0xB0, 0x00, 0x02, b'\x00\x80')  # Channel 3: +0.5V read
        
    def set_analog_mode(self, enable=True):
        """Enable/disable analog SRM operation mode"""
        if not self.connected:
            raise RuntimeError("FPGA not connected")
        cmd = b'\x01' if enable else b'\x00'
        self._send_control_packet(0xA4, 0x00, 0x00, cmd)
        self.analog_mode_enabled = enable
        
    def configure_noise_profile(self, noise_level):
        """Configure analog noise profile (0.0-1.0 scale)"""
        if not self.connected:
            raise RuntimeError("FPGA not connected")
        level = int(noise_level * 255).to_bytes(1, 'big')
        self._send_control_packet(0xA5, 0x00, 0x00, level)
        
    def supports_analog(self):
        """Check if analog mode is supported"""
        return self.connected and self.analog_mode_enabled
        
    def program_layer_weights(self, layer_name, weights):
        """Program 32x32 crossbar array for specific layer using block-wise updates"""
        if not self.connected:
            raise RuntimeError("FPGA not connected")
        
        # Split large weight matrices into 32x32 blocks
        weight_blocks = self._split_into_blocks(weights)
        
        # Program each block sequentially
        for block_idx, block in enumerate(weight_blocks):
            quantized = self._quantize_weights(block)
            matrix = self._reshape_to_crossbar(quantized)
            
            # Send programming sequence
            layer_code = self.layer_mapping[layer_name]
            self._send_control_packet(0xA1, layer_code, block_idx, b'PROG_START')
            for row in range(32):
                row_data = matrix[row].tobytes()
                self._send_bulk_data(0x02, row_data)
                self._send_control_packet(0xA2, row, block_idx, b'ROW_DATA')
            self._send_control_packet(0xA3, block_idx, 0x00, b'PROG_END')
        
    def _split_into_blocks(self, tensor):
        """Split weight tensor into 32x32 blocks"""
        tensor = tensor.cpu().numpy()
        rows, cols = tensor.shape[-2], tensor.shape[-1]
        blocks = []
        
        for i in range(0, rows, 32):
            for j in range(0, cols, 32):
                block = tensor[..., i:i+32, j:j+32]
                if block.size < 1024:
                    block = np.pad(block, ((0,0),(0,32-block.shape[-1])))
                blocks.append(block)
        return blocks
        
    def _quantize_weights(self, weights):
        """Convert weights to 16-bit DAC format (-32768 to +32767)"""
        scaled = np.clip(weights * 32767, -32768, 32767)
        return scaled.astype(np.int16)
    
    def _reshape_to_crossbar(self, data):
        """Reshape data to 32x32 crossbar format with hardware padding"""
        if data.size > 1024:
            raise ValueError("Weight block exceeds 32x32 capacity")
        return np.pad(data.flat[:1024], (0, 1024-data.size)).reshape(32,32)
    
    def _send_control_packet(self, request, value, index, data, timeout=1000):
        """Send USB control packet with FTDI protocol"""
        try:
            self.dev.ctrl_transfer(
                0x40,    # Request type
                request,  # Request code
                value,    # Value
                index,    # Index
                data,     # Data payload
                timeout)
        except Exception as e:
            raise RuntimeError(f"Control packet failed: {str(e)}")
            
    def _send_bulk_data(self, endpoint, data):
        """Send bulk data to specified endpoint"""
        try:
            self.dev.write(endpoint, data, timeout=5000)
        except Exception as e:
            raise RuntimeError(f"Bulk transfer failed: {str(e)}")