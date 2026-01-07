"""
Evaluation Test Set Generator
=============================
Genera test set separato per valutazione finale (non visto in training).
"""

import json
import random
from pathlib import Path
from typing import List, Dict


def generate_evaluation_set(output_dir: str = "evaluation") -> Dict:
    """
    Genera test set per evaluation finale.
    
    IMPORTANTE: Questi esempi NON devono essere usati in training!
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    test_sets = {}
    
    # 1. Arithmetic Test (100 problems)
    arithmetic_test = []
    for _ in range(100):
        op = random.choice(['+', '-', '*', '/'])
        if op == '+':
            a, b = random.randint(100, 9999), random.randint(100, 9999)
            answer = a + b
            problem = f"{a} + {b}"
        elif op == '-':
            a = random.randint(1000, 9999)
            b = random.randint(100, a)
            answer = a - b
            problem = f"{a} - {b}"
        elif op == '*':
            a, b = random.randint(10, 99), random.randint(10, 99)
            answer = a * b
            problem = f"{a} × {b}"
        else:
            b = random.randint(2, 50)
            answer = random.randint(10, 100)
            a = b * answer
            problem = f"{a} ÷ {b}"
        
        arithmetic_test.append({
            "id": f"arith_{len(arithmetic_test)+1}",
            "problem": f"Calculate: {problem}",
            "expected_answer": str(answer),
            "category": "arithmetic"
        })
    
    test_sets["arithmetic"] = arithmetic_test
    
    # 2. Algebra Test (50 problems)
    algebra_test = []
    for i in range(50):
        a = random.randint(2, 15)
        x_sol = random.randint(-15, 15)
        b = random.randint(-30, 30)
        c = a * x_sol + b
        
        algebra_test.append({
            "id": f"alg_{i+1}",
            "problem": f"Solve for x: {a}x + {b} = {c}",
            "expected_answer": str(x_sol),
            "category": "algebra"
        })
    
    test_sets["algebra"] = algebra_test
    
    # 3. Code Comprehension Test (50 problems)
    code_templates = [
        {
            "code": "def double(x): return x * 2",
            "gen_test": lambda: (random.randint(1, 100), lambda x: x * 2)
        },
        {
            "code": "def square(x): return x ** 2",
            "gen_test": lambda: (random.randint(1, 20), lambda x: x ** 2)
        },
        {
            "code": "def add_ten(x): return x + 10",
            "gen_test": lambda: (random.randint(1, 100), lambda x: x + 10)
        },
        {
            "code": "def is_even(x): return x % 2 == 0",
            "gen_test": lambda: (random.randint(1, 100), lambda x: x % 2 == 0)
        },
    ]
    
    code_test = []
    for i in range(50):
        template = random.choice(code_templates)
        input_val, fn = template["gen_test"]()
        expected = fn(input_val)
        
        code_test.append({
            "id": f"code_{i+1}",
            "problem": f"Given:\n{template['code']}\n\nWhat is the output of the function when called with {input_val}?",
            "expected_answer": str(expected),
            "category": "code"
        })
    
    test_sets["code"] = code_test
    
    # 4. Word Problems (25 problems)
    word_problems = []
    for i in range(25):
        scenario = random.choice([
            lambda: {
                "text": f"Alice has {(a := random.randint(5, 20))} apples. Bob gives her {(b := random.randint(5, 15))} more. How many apples does Alice have now?",
                "answer": a + b
            },
            lambda: {
                "text": f"A store has {(a := random.randint(50, 100))} items. They sell {(b := random.randint(10, a-10))} items. How many items remain?",
                "answer": a - b
            },
            lambda: {
                "text": f"Each box contains {(a := random.randint(5, 12))} pencils. How many pencils are in {(b := random.randint(3, 10))} boxes?",
                "answer": a * b
            },
        ])
        
        s = scenario()
        word_problems.append({
            "id": f"word_{i+1}",
            "problem": s["text"],
            "expected_answer": str(s["answer"]),
            "category": "word_problem"
        })
    
    test_sets["word_problems"] = word_problems
    
    # Save all test sets
    for name, tests in test_sets.items():
        filepath = output_path / f"{name}_test.json"
        with open(filepath, 'w') as f:
            json.dump(tests, f, indent=2)
        print(f"Saved {len(tests)} tests to {filepath}")
    
    # Combined test set
    all_tests = []
    for tests in test_sets.values():
        all_tests.extend(tests)
    
    random.shuffle(all_tests)
    
    with open(output_path / "combined_test.json", 'w') as f:
        json.dump(all_tests, f, indent=2)
    
    stats = {
        "total_tests": len(all_tests),
        "categories": {name: len(tests) for name, tests in test_sets.items()},
        "note": "DO NOT USE IN TRAINING - EVALUATION ONLY"
    }
    
    with open(output_path / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ Generated {len(all_tests)} evaluation tests")
    print(f"  Output: {output_path}")
    
    return stats


if __name__ == "__main__":
    print("=== Evaluation Test Set Generator ===\n")
    generate_evaluation_set()
