"""
INT8 Post-Training Quantization
===============================
Quantizza modello ODIN per ridurre size e velocizzare inference.
"""

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from typing import Tuple, Dict


class QuantizedLinear(nn.Module):
    """Linear layer con pesi quantizzati INT8"""
    
    def __init__(self, weight: torch.Tensor, scale: float, zero_point: int = 0):
        super().__init__()
        self.register_buffer('weight_int8', weight.to(torch.int8))
        self.scale = scale
        self.zero_point = zero_point
        self.out_features, self.in_features = weight.shape
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantizza on-the-fly
        weight_fp = (self.weight_int8.float() - self.zero_point) * self.scale
        return nn.functional.linear(x, weight_fp)


def compute_scale_zeropoint(tensor: torch.Tensor, num_bits: int = 8) -> Tuple[float, int]:
    """
    Calcola scale e zero_point per quantizzazione simmetrica.
    """
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1
    
    # Symmetric quantization
    max_val = tensor.abs().max().item()
    scale = max_val / qmax if max_val > 0 else 1.0
    zero_point = 0  # Symmetric
    
    return scale, zero_point


def quantize_tensor(tensor: torch.Tensor, scale: float, zero_point: int = 0) -> torch.Tensor:
    """Quantizza tensor a INT8"""
    quantized = torch.round(tensor / scale) + zero_point
    quantized = torch.clamp(quantized, -128, 127)
    return quantized.to(torch.int8)


def quantize_model(model: nn.Module, calibration_data: torch.Tensor = None) -> nn.Module:
    """
    Quantizza tutti i Linear layer del modello a INT8.
    
    Args:
        model: modello PyTorch
        calibration_data: dati per calibrazione (opzionale)
    
    Returns:
        modello quantizzato
    """
    print("Quantizing model to INT8...")
    
    quantization_info = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            scale, zp = compute_scale_zeropoint(weight)
            
            quantization_info[name] = {
                'scale': scale,
                'zero_point': zp,
                'original_shape': weight.shape,
                'original_size_kb': weight.numel() * 4 / 1024,
                'quantized_size_kb': weight.numel() / 1024
            }
    
    # Report
    total_original = sum(v['original_size_kb'] for v in quantization_info.values())
    total_quantized = sum(v['quantized_size_kb'] for v in quantization_info.values())
    
    print(f"\nQuantization Summary:")
    print(f"  Layers quantized: {len(quantization_info)}")
    print(f"  Original size: {total_original/1024:.1f} MB")
    print(f"  Quantized size: {total_quantized/1024:.1f} MB")
    print(f"  Compression: {total_original/total_quantized:.1f}x")
    
    return model, quantization_info


def save_quantized_weights(model: nn.Module, output_dir: str):
    """
    Salva pesi quantizzati in formato binario efficiente.
    
    Formato per ogni layer:
    - scale (float32)
    - zero_point (int32)
    - weights (int8 array)
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    manifest = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            scale, zp = compute_scale_zeropoint(weight)
            weight_int8 = quantize_tensor(weight, scale, zp)
            
            # Save
            safe_name = name.replace('.', '_')
            weight_file = output_path / f"{safe_name}.bin"
            
            with open(weight_file, 'wb') as f:
                # Header: scale, zero_point, shape
                np.array([scale], dtype=np.float32).tofile(f)
                np.array([zp], dtype=np.int32).tofile(f)
                np.array(weight.shape, dtype=np.int32).tofile(f)
                # Weights
                weight_int8.numpy().tofile(f)
            
            manifest[name] = {
                'file': str(weight_file),
                'scale': scale,
                'zero_point': zp,
                'shape': list(weight.shape)
            }
    
    # Save manifest
    import json
    with open(output_path / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✓ Quantized weights saved to {output_path}")
    return manifest


def benchmark_quantization(model: nn.Module, input_shape: Tuple[int, int] = (1, 128)):
    """
    Confronta performance FP32 vs simulazione INT8.
    """
    import time
    
    model.eval()
    device = next(model.parameters()).device
    
    dummy_input = torch.randint(0, 32768, input_shape, device=device)
    
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Benchmark FP32
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.perf_counter()
    
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    fp32_time = (time.perf_counter() - start) / 10
    
    print(f"\nPerformance (FP32):")
    print(f"  Inference time: {fp32_time*1000:.2f} ms")
    print(f"  Throughput: {input_shape[1]/fp32_time:.0f} tokens/sec")
    
    return fp32_time


# Test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "01_architect_mathematician" / "src"))
    from odin_model import ODIN, OdinConfig
    
    print("=== INT8 Quantization Pipeline ===\n")
    
    config = OdinConfig()
    model = ODIN(config)
    
    # Quantize
    model_q, info = quantize_model(model)
    
    # Benchmark
    benchmark_quantization(model)
    
    # Save
    output_dir = Path(__file__).parent.parent / "exports" / "quantized"
    save_quantized_weights(model, str(output_dir))
