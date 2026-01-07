"""
Browser Memory Optimizer
========================
Ottimizzazioni per ridurre memory footprint nel browser.
"""

import struct
from pathlib import Path
from typing import Dict, List, Tuple
import json


class MemoryOptimizer:
    """
    Ottimizza modello per memory-constrained browser environment.
    
    Target: < 400MB total memory usage
    """
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.stats = {}
    
    def analyze_model(self) -> Dict:
        """Analizza memory footprint del modello"""
        
        # Simulated analysis (in prod, parse actual ONNX)
        config = {
            "vocab_size": 32768,
            "embedding_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "ffn_dim": 2688
        }
        
        # Calculate sizes
        embedding_params = config["vocab_size"] * config["embedding_dim"]
        
        per_layer_params = (
            # Time-mixing
            4 * config["embedding_dim"] ** 2 +  # W_r, W_k, W_v, W_o
            3 * config["embedding_dim"] +        # time_mix_r/k/v
            2 * config["num_heads"] * (config["embedding_dim"] // config["num_heads"]) +  # decay, first
            # Channel-mixing
            2 * config["embedding_dim"] * config["ffn_dim"] +  # W_k, W_v
            config["embedding_dim"] ** 2 +  # W_r
            2 * config["embedding_dim"] +   # time_mix_k/r
            # LayerNorms
            2 * config["embedding_dim"]     # ln weights
        )
        
        total_params = embedding_params + config["num_layers"] * per_layer_params
        
        self.stats = {
            "total_params": total_params,
            "embedding_params": embedding_params,
            "per_layer_params": per_layer_params,
            "fp32_size_mb": total_params * 4 / 1e6,
            "fp16_size_mb": total_params * 2 / 1e6,
            "int8_size_mb": total_params / 1e6,
            "int4_size_mb": total_params * 0.5 / 1e6,
        }
        
        return self.stats
    
    def estimate_runtime_memory(self, batch_size: int = 1, seq_len: int = 512) -> Dict:
        """Stima memoria runtime (weights + activations + state)"""
        
        config = {
            "embedding_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "vocab_size": 32768,
            "ffn_dim": 2688
        }
        
        # Weights (INT8)
        weights_mb = self.stats.get("int8_size_mb", 100)
        
        # Activations (FP32 for computation)
        # Per layer: 2 * batch * seq * dim (input + output of each block)
        activations_per_layer = 2 * batch_size * seq_len * config["embedding_dim"] * 4 / 1e6
        activations_mb = config["num_layers"] * activations_per_layer
        
        # RNN State (persistent between tokens)
        # Per layer: batch * num_heads * head_dim * 3 (num, den, prev_x)
        head_dim = config["embedding_dim"] // config["num_heads"]
        state_per_layer = batch_size * config["num_heads"] * head_dim * 3 * 4 / 1e6
        state_mb = config["num_layers"] * state_per_layer
        
        # Output logits buffer
        logits_mb = batch_size * config["vocab_size"] * 4 / 1e6
        
        # Total
        total_mb = weights_mb + activations_mb + state_mb + logits_mb
        
        return {
            "weights_mb": weights_mb,
            "activations_mb": activations_mb,
            "state_mb": state_mb,
            "logits_mb": logits_mb,
            "total_mb": total_mb,
            "under_budget": total_mb < 400
        }
    
    def generate_memory_layout(self, output_path: str) -> Dict:
        """Genera layout memoria ottimizzato per WASM"""
        
        layout = {
            "version": 1,
            "memory_pages": 6400,  # 400MB / 64KB per page
            "sections": [
                {
                    "name": "weights",
                    "offset": 0,
                    "size_pages": 1600,  # ~100MB for INT8 weights
                    "type": "static"
                },
                {
                    "name": "activations",
                    "offset": 1600,
                    "size_pages": 3200,  # ~200MB for activations
                    "type": "dynamic"
                },
                {
                    "name": "state",
                    "offset": 4800,
                    "size_pages": 800,   # ~50MB for RNN state
                    "type": "persistent"
                },
                {
                    "name": "scratch",
                    "offset": 5600,
                    "size_pages": 800,   # ~50MB scratch space
                    "type": "temporary"
                }
            ],
            "optimization_flags": {
                "reuse_activation_buffers": True,
                "stream_weights": False,  # All weights in memory
                "quantized_activations": False,  # Keep FP32 for quality
                "lazy_logits": True  # Only compute needed logits
            }
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(layout, f, indent=2)
        
        return layout
    
    def optimize_weight_loading(self) -> Dict:
        """Strategie per loading efficiente pesi"""
        
        strategies = {
            "chunked_loading": {
                "description": "Load weights in chunks to avoid memory spikes",
                "chunk_size_mb": 10,
                "parallel_chunks": 4
            },
            "progressive_loading": {
                "description": "Load embeddings first, then layers sequentially",
                "order": ["embedding", "ln_in", "blocks.0-3", "blocks.4-7", "blocks.8-11", "ln_out", "head"]
            },
            "compression": {
                "description": "Compress weights for transfer, decompress in WASM",
                "format": "gzip",
                "expected_ratio": 0.7
            },
            "caching": {
                "description": "Cache in IndexedDB after first load",
                "cache_key": "odin-100m-weights-v1",
                "expiry_days": 30
            }
        }
        
        return strategies


def print_optimization_report():
    """Genera report ottimizzazioni"""
    
    optimizer = MemoryOptimizer("model.onnx")
    
    print("=" * 60)
    print("ODIN Browser Memory Optimization Report")
    print("=" * 60)
    
    # Model analysis
    stats = optimizer.analyze_model()
    print(f"\n📊 Model Size:")
    print(f"  Total parameters: {stats['total_params']/1e6:.1f}M")
    print(f"  FP32: {stats['fp32_size_mb']:.1f} MB")
    print(f"  FP16: {stats['fp16_size_mb']:.1f} MB")
    print(f"  INT8: {stats['int8_size_mb']:.1f} MB (recommended)")
    print(f"  INT4: {stats['int4_size_mb']:.1f} MB (experimental)")
    
    # Runtime memory
    runtime = optimizer.estimate_runtime_memory(batch_size=1, seq_len=512)
    print(f"\n🧠 Runtime Memory (batch=1, seq=512):")
    print(f"  Weights: {runtime['weights_mb']:.1f} MB")
    print(f"  Activations: {runtime['activations_mb']:.1f} MB")
    print(f"  State: {runtime['state_mb']:.1f} MB")
    print(f"  Logits: {runtime['logits_mb']:.1f} MB")
    print(f"  ────────────────────")
    print(f"  Total: {runtime['total_mb']:.1f} MB")
    print(f"  Budget (400MB): {'✓ OK' if runtime['under_budget'] else '✗ EXCEEDED'}")
    
    # Strategies
    strategies = optimizer.optimize_weight_loading()
    print(f"\n⚡ Loading Strategies:")
    for name, strategy in strategies.items():
        print(f"  • {name}: {strategy['description']}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_optimization_report()
