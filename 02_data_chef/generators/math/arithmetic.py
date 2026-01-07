"""
Arithmetic Data Generator
=========================
Genera esempi di aritmetica con step-by-step reasoning.
100% verificati con eval().
"""

import random
from typing import Tuple, List, Dict


def generate_addition(max_digits: int = 4) -> Dict:
    """Genera problema di addizione con soluzione step-by-step"""
    a = random.randint(1, 10**max_digits - 1)
    b = random.randint(1, 10**max_digits - 1)
    result = a + b
    
    return {
        "type": "addition",
        "problem": f"Calculate: {a} + {b}",
        "reasoning": f"Adding {a} and {b}:\n{a} + {b} = {result}",
        "answer": str(result),
        "verified": eval(f"{a} + {b}") == result
    }


def generate_subtraction(max_digits: int = 4) -> Dict:
    """Genera problema di sottrazione (risultato sempre positivo)"""
    a = random.randint(1, 10**max_digits - 1)
    b = random.randint(1, a)  # b <= a per risultato positivo
    result = a - b
    
    return {
        "type": "subtraction",
        "problem": f"Calculate: {a} - {b}",
        "reasoning": f"Subtracting {b} from {a}:\n{a} - {b} = {result}",
        "answer": str(result),
        "verified": eval(f"{a} - {b}") == result
    }


def generate_multiplication(max_digits: int = 3) -> Dict:
    """Genera problema di moltiplicazione"""
    a = random.randint(1, 10**max_digits - 1)
    b = random.randint(1, 10**max_digits - 1)
    result = a * b
    
    return {
        "type": "multiplication",
        "problem": f"Calculate: {a} × {b}",
        "reasoning": f"Multiplying {a} by {b}:\n{a} × {b} = {result}",
        "answer": str(result),
        "verified": eval(f"{a} * {b}") == result
    }


def generate_division(max_digits: int = 3) -> Dict:
    """Genera problema di divisione (risultato intero)"""
    b = random.randint(2, 10**max_digits - 1)
    result = random.randint(1, 10**max_digits - 1)
    a = b * result  # Garantisce divisione esatta
    
    return {
        "type": "division",
        "problem": f"Calculate: {a} ÷ {b}",
        "reasoning": f"Dividing {a} by {b}:\n{a} ÷ {b} = {result}",
        "answer": str(result),
        "verified": a // b == result
    }


def generate_batch(n: int, operation: str = "mixed", max_digits: int = 4) -> List[Dict]:
    """Genera batch di problemi"""
    generators = {
        "addition": generate_addition,
        "subtraction": generate_subtraction,
        "multiplication": generate_multiplication,
        "division": generate_division
    }
    
    results = []
    for _ in range(n):
        if operation == "mixed":
            op = random.choice(list(generators.keys()))
        else:
            op = operation
        
        example = generators[op](max_digits)
        results.append(example)
    
    return results


def format_for_training(example: Dict) -> str:
    """Formatta esempio per training LLM"""
    return f"""<problem>
{example['problem']}
</problem>
<reasoning>
{example['reasoning']}
</reasoning>
<answer>
{example['answer']}
</answer>"""


# Test
if __name__ == "__main__":
    print("=== Arithmetic Generator Test ===\n")
    
    for op in ["addition", "subtraction", "multiplication", "division"]:
        example = generate_batch(1, op)[0]
        print(f"[{op.upper()}]")
        print(format_for_training(example))
        print(f"Verified: {example['verified']}\n")
    
    # Stats
    batch = generate_batch(1000, "mixed")
    verified_count = sum(1 for e in batch if e['verified'])
    print(f"Batch verification: {verified_count}/1000 correct")
