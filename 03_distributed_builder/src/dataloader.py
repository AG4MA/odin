"""
Synthetic Dataset Loader
========================
Carica e tokenizza dati sintetici per training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from typing import List, Dict, Optional


class SyntheticMathDataset(Dataset):
    """Dataset per problemi matematici sintetici"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        split: str = "train"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_path, split)
    
    def _load_data(self, data_path: str, split: str) -> List[Dict]:
        """Carica dati da file JSON/JSONL"""
        path = Path(data_path)
        
        if path.suffix == ".jsonl":
            data = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
        else:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.data[idx]
        
        # Formatta testo
        text = self._format_example(example)
        
        # Tokenizza
        tokens = self.tokenizer.encode(text)
        
        # Padding/truncation
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        return {
            "input_ids": tokens,
            "targets": tokens.clone()  # Per autoregressive LM
        }
    
    def _format_example(self, example: Dict) -> str:
        """Formatta esempio in testo"""
        if "problem" in example and "reasoning" in example:
            return f"""<problem>
{example['problem']}
</problem>
<reasoning>
{example['reasoning']}
</reasoning>
<answer>
{example['answer']}
</answer>"""
        else:
            return example.get("text", str(example))


class SimpleTokenizer:
    """Tokenizer semplice per prototipazione (sostituire con BPE)"""
    
    def __init__(self, vocab_size: int = 32768):
        self.vocab_size = vocab_size
        self.char_to_id = {}
        self.id_to_char = {}
        
        # Basic ASCII + special tokens
        special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
        for i, tok in enumerate(special_tokens):
            self.char_to_id[tok] = i
            self.id_to_char[i] = tok
        
        offset = len(special_tokens)
        for i in range(256):
            char = chr(i) if 32 <= i < 127 else f"<byte_{i}>"
            self.char_to_id[char] = i + offset
            self.id_to_char[i + offset] = char
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token ids"""
        ids = [self.char_to_id.get("<bos>", 2)]
        for char in text:
            if char in self.char_to_id:
                ids.append(self.char_to_id[char])
            else:
                ids.append(self.char_to_id.get("<unk>", 1))
        ids.append(self.char_to_id.get("<eos>", 3))
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """Decode token ids to text"""
        chars = []
        for id in ids:
            if id in self.id_to_char:
                token = self.id_to_char[id]
                if not token.startswith("<"):
                    chars.append(token)
        return "".join(chars)


def create_dataloader(
    data_path: str,
    tokenizer,
    batch_size: int = 32,
    max_length: int = 512,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """Crea DataLoader per training"""
    
    dataset = SyntheticMathDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


# Test
if __name__ == "__main__":
    print("=== DataLoader Test ===\n")
    
    # Test tokenizer
    tokenizer = SimpleTokenizer()
    
    text = "Calculate: 2 + 3 = 5"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {text}")
    print(f"Encoded: {encoded[:20]}...")
    print(f"Decoded: {decoded}")
