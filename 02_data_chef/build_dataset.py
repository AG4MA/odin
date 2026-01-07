"""
Dataset Builder & Packager
==========================
Assembla, valida e pacchettizza il dataset finale per training.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Generator, Any
import sys

# Import generators
sys.path.insert(0, str(Path(__file__).parent))
from math.arithmetic import generate_batch as gen_arithmetic
from math.algebra import generate_batch as gen_algebra
from code.python_basic import generate_batch as gen_python


def generate_full_dataset(
    num_arithmetic: int = 500000,
    num_algebra: int = 300000,
    num_python: int = 200000,
    output_dir: str = "dataset"
) -> Dict:
    """
    Genera dataset completo per ODIN-100M.
    
    Target: ~1M esempi = ~10B tokens circa
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    stats: Dict[str, Any] = {
        "arithmetic": 0,
        "algebra": 0, 
        "python": 0,
        "total": 0,
        "verified": 0
    }
    
    # Generate in chunks to manage memory
    chunk_size = 10000
    
    print("Generating Synthetic Dataset...")
    print("=" * 50)
    
    all_examples = []
    
    # Arithmetic
    print(f"\n[1/3] Arithmetic: {num_arithmetic} examples")
    for i in range(0, num_arithmetic, chunk_size):
        batch = gen_arithmetic(min(chunk_size, num_arithmetic - i), "mixed")
        all_examples.extend(batch)
        stats["arithmetic"] += len(batch)
        stats["verified"] += sum(1 for e in batch if e.get("verified", False))
        print(f"  Progress: {stats['arithmetic']}/{num_arithmetic}")
    
    # Algebra
    print(f"\n[2/3] Algebra: {num_algebra} examples")
    for i in range(0, num_algebra, chunk_size):
        batch = gen_algebra(min(chunk_size, num_algebra - i), "mixed")
        all_examples.extend(batch)
        stats["algebra"] += len(batch)
        stats["verified"] += sum(1 for e in batch if e.get("verified", False))
        print(f"  Progress: {stats['algebra']}/{num_algebra}")
    
    # Python
    print(f"\n[3/3] Python Code: {num_python} examples")
    for i in range(0, num_python, chunk_size):
        batch = gen_python(min(chunk_size, num_python - i))
        all_examples.extend(batch)
        stats["python"] += len(batch)
        stats["verified"] += sum(1 for e in batch if e.get("verified", False))
        print(f"  Progress: {stats['python']}/{num_python}")
    
    stats["total"] = len(all_examples)
    
    # Shuffle
    print("\nShuffling dataset...")
    random.shuffle(all_examples)
    
    # Split train/val/test
    n = len(all_examples)
    train_end = int(n * 0.95)
    val_end = int(n * 0.98)
    
    splits = {
        "train": all_examples[:train_end],
        "validation": all_examples[train_end:val_end],
        "test": all_examples[val_end:]
    }
    
    # Save
    print("\nSaving splits...")
    for split_name, data in splits.items():
        filepath = output_path / f"{split_name}.jsonl"
        with open(filepath, 'w', encoding='utf-8') as f:
            for example in data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        print(f"  {split_name}: {len(data)} examples → {filepath}")
    
    # Save stats
    stats["splits"] = {k: len(v) for k, v in splits.items()}
    stats["verification_rate"] = stats["verified"] / stats["total"] * 100
    
    with open(output_path / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "=" * 50)
    print("Dataset Generation Complete!")
    print(f"  Total examples: {stats['total']}")
    print(f"  Verification rate: {stats['verification_rate']:.1f}%")
    print(f"  Output: {output_path}")
    
    return stats


def estimate_tokens(dataset_path: str) -> Dict:
    """Stima numero di token nel dataset"""
    
    total_chars = 0
    total_examples = 0
    
    for split in ["train", "validation", "test"]:
        filepath = Path(dataset_path) / f"{split}.jsonl"
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    example = json.loads(line)
                    text = format_example(example)
                    total_chars += len(text)
                    total_examples += 1
    
    # Rough estimate: ~4 chars per token
    estimated_tokens = total_chars / 4
    
    return {
        "total_chars": total_chars,
        "total_examples": total_examples,
        "estimated_tokens": estimated_tokens,
        "estimated_tokens_M": estimated_tokens / 1e6,
        "estimated_tokens_B": estimated_tokens / 1e9
    }


def format_example(example: Dict) -> str:
    """Formatta esempio in testo per training"""
    if "code" in example.get("answer", "") or "def " in str(example.get("answer", "")):
        return f"""<problem>
{example.get('problem', '')}
</problem>
<reasoning>
{example.get('reasoning', '')}
</reasoning>
<code>
{example.get('answer', '')}
</code>"""
    else:
        return f"""<problem>
{example.get('problem', '')}
</problem>
<reasoning>
{example.get('reasoning', '')}
</reasoning>
<answer>
{example.get('answer', '')}
</answer>"""


# Test
if __name__ == "__main__":
    # Small test dataset
    print("=== Dataset Builder Test ===\n")
    
    stats = generate_full_dataset(
        num_arithmetic=100,
        num_algebra=100,
        num_python=50,
        output_dir="test_dataset"
    )
    
    # Estimate tokens
    token_stats = estimate_tokens("test_dataset")
    print(f"\nToken Estimate:")
    print(f"  Estimated tokens: {token_stats['estimated_tokens_M']:.2f}M")
