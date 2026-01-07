"""
ONNX Export Pipeline
====================
Esporta modello PyTorch a ONNX per conversione WASM.
"""

import torch
import torch.onnx
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "01_architect_mathematician" / "src"))


def export_to_onnx(
    model,
    output_path: str,
    seq_len: int = 128,
    batch_size: int = 1,
    opset_version: int = 14
):
    """
    Esporta modello ODIN a ONNX.
    
    Args:
        model: ODIN model instance
        output_path: percorso file .onnx
        seq_len: lunghezza sequenza per export
        batch_size: batch size per export
        opset_version: versione ONNX opset
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Dummy input
    dummy_input = torch.randint(
        0, model.config.vocab_size, 
        (batch_size, seq_len),
        device=device
    )
    
    print(f"Exporting model to {output_path}")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Opset version: {opset_version}")
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size', 1: 'sequence_length'}
        }
    )
    
    print(f"✓ Export completed: {output_path}")
    
    # Validate
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model validated")
    
    return output_path


def optimize_onnx(input_path: str, output_path: str = None):
    """
    Ottimizza grafo ONNX.
    
    - Constant folding
    - Dead code elimination
    - Operator fusion
    """
    import onnx
    from onnx import optimizer
    
    if output_path is None:
        output_path = input_path.replace('.onnx', '_optimized.onnx')
    
    print(f"Optimizing ONNX model...")
    
    model = onnx.load(input_path)
    
    # Ottimizzazioni
    passes = [
        'eliminate_identity',
        'eliminate_nop_transpose',
        'eliminate_deadend',
        'fuse_consecutive_transposes',
        'fuse_matmul_add_bias_into_gemm',
        'fuse_bn_into_conv',
    ]
    
    try:
        optimized = optimizer.optimize(model, passes)
        onnx.save(optimized, output_path)
        print(f"✓ Optimized model saved: {output_path}")
    except Exception as e:
        print(f"⚠ Optimization failed, saving original: {e}")
        onnx.save(model, output_path)
    
    # Size comparison
    import os
    original_size = os.path.getsize(input_path) / 1e6
    optimized_size = os.path.getsize(output_path) / 1e6
    print(f"  Original: {original_size:.1f} MB")
    print(f"  Optimized: {optimized_size:.1f} MB")
    
    return output_path


def verify_onnx_output(pytorch_model, onnx_path: str, atol: float = 1e-4):
    """
    Verifica che output ONNX corrisponda a PyTorch.
    """
    import onnxruntime as ort
    import numpy as np
    
    pytorch_model.eval()
    device = next(pytorch_model.parameters()).device
    
    # Random input
    input_ids = torch.randint(0, pytorch_model.config.vocab_size, (1, 64), device=device)
    
    # PyTorch output
    with torch.no_grad():
        pt_output, _ = pytorch_model(input_ids)
    pt_output = pt_output.cpu().numpy()
    
    # ONNX output
    session = ort.InferenceSession(onnx_path)
    ort_output = session.run(
        None, 
        {'input_ids': input_ids.cpu().numpy()}
    )[0]
    
    # Compare
    diff = np.abs(pt_output - ort_output).max()
    print(f"Max difference: {diff}")
    
    if diff < atol:
        print("✓ Outputs match within tolerance")
        return True
    else:
        print("✗ Outputs differ significantly!")
        return False


# Main
if __name__ == "__main__":
    from odin_model import ODIN, OdinConfig
    
    print("=== ONNX Export Pipeline ===\n")
    
    # Create model
    config = OdinConfig()
    model = ODIN(config)
    
    # Export
    output_dir = Path(__file__).parent.parent / "exports"
    output_dir.mkdir(exist_ok=True)
    
    onnx_path = str(output_dir / "odin_100m.onnx")
    export_to_onnx(model, onnx_path)
    
    # Verify
    print("\nVerifying export...")
    verify_onnx_output(model, onnx_path)
