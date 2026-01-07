"""
Reasoning Benchmark Suite
=========================
Valuta capacità di reasoning del modello ODIN-100M.
"""

import json
import random
from typing import Dict, List, Tuple
from pathlib import Path


class ReasoningBenchmark:
    """
    Benchmark suite per valutare reasoning capabilities.
    
    Categories:
    - Arithmetic: operazioni matematiche base
    - Algebra: equazioni e problem solving
    - Logic: ragionamento logico
    - Code: comprensione e generazione codice
    """
    
    def __init__(self):
        self.results = {}
        self.benchmarks = self._create_benchmarks()
    
    def _create_benchmarks(self) -> Dict[str, List[Dict]]:
        """Crea suite di benchmark"""
        
        return {
            "arithmetic": self._gen_arithmetic_bench(100),
            "algebra": self._gen_algebra_bench(50),
            "logic": self._gen_logic_bench(50),
            "code": self._gen_code_bench(50),
        }
    
    def _gen_arithmetic_bench(self, n: int) -> List[Dict]:
        """Benchmark aritmetica"""
        problems = []
        
        for _ in range(n // 4):
            # Addition
            a, b = random.randint(10, 999), random.randint(10, 999)
            problems.append({
                "problem": f"Calculate: {a} + {b}",
                "answer": str(a + b),
                "category": "arithmetic",
                "difficulty": "easy"
            })
            
            # Subtraction
            a, b = random.randint(100, 999), random.randint(10, 99)
            problems.append({
                "problem": f"Calculate: {a} - {b}",
                "answer": str(a - b),
                "category": "arithmetic",
                "difficulty": "easy"
            })
            
            # Multiplication
            a, b = random.randint(10, 99), random.randint(2, 20)
            problems.append({
                "problem": f"Calculate: {a} × {b}",
                "answer": str(a * b),
                "category": "arithmetic",
                "difficulty": "medium"
            })
            
            # Division
            b = random.randint(2, 20)
            result = random.randint(5, 50)
            a = b * result
            problems.append({
                "problem": f"Calculate: {a} ÷ {b}",
                "answer": str(result),
                "category": "arithmetic",
                "difficulty": "medium"
            })
        
        return problems
    
    def _gen_algebra_bench(self, n: int) -> List[Dict]:
        """Benchmark algebra"""
        problems = []
        
        for _ in range(n):
            a = random.randint(2, 10)
            x_sol = random.randint(-10, 10)
            b = random.randint(-20, 20)
            c = a * x_sol + b
            
            if b >= 0:
                prob = f"Solve for x: {a}x + {b} = {c}"
            else:
                prob = f"Solve for x: {a}x - {abs(b)} = {c}"
            
            problems.append({
                "problem": prob,
                "answer": f"x = {x_sol}",
                "category": "algebra",
                "difficulty": "medium"
            })
        
        return problems
    
    def _gen_logic_bench(self, n: int) -> List[Dict]:
        """Benchmark logica"""
        problems = []
        
        syllogisms = [
            ("All cats are animals. Whiskers is a cat.", "Whiskers is an animal.", True),
            ("All birds can fly. Penguins are birds.", "Penguins can fly.", False),
            ("If it rains, the ground is wet. It is raining.", "The ground is wet.", True),
            ("All squares are rectangles. This shape is a square.", "This shape is a rectangle.", True),
        ]
        
        for _ in range(n):
            premise, conclusion, is_valid = random.choice(syllogisms)
            problems.append({
                "problem": f"Premise: {premise}\nConclusion: {conclusion}\nIs this conclusion logically valid?",
                "answer": "Yes" if is_valid else "No",
                "category": "logic",
                "difficulty": "medium"
            })
        
        return problems
    
    def _gen_code_bench(self, n: int) -> List[Dict]:
        """Benchmark code comprehension"""
        problems = []
        
        code_problems = [
            ("def f(x): return x * 2", "f(5)", "10"),
            ("def f(a, b): return a + b", "f(3, 4)", "7"),
            ("def f(n): return n ** 2", "f(6)", "36"),
            ("def f(s): return len(s)", 'f("hello")', "5"),
            ("def f(lst): return sum(lst)", "f([1, 2, 3])", "6"),
        ]
        
        for _ in range(n):
            code, call, result = random.choice(code_problems)
            problems.append({
                "problem": f"Given the function:\n{code}\n\nWhat is the output of {call}?",
                "answer": result,
                "category": "code",
                "difficulty": "easy"
            })
        
        return problems
    
    def evaluate(self, model, tokenizer) -> Dict:
        """
        Esegue valutazione su tutti i benchmark.
        
        Args:
            model: ODIN model
            tokenizer: tokenizer instance
        
        Returns:
            Dict con risultati per categoria
        """
        results = {}
        
        for category, problems in self.benchmarks.items():
            correct = 0
            total = len(problems)
            
            for prob in problems:
                # Generate response
                prompt = f"<problem>\n{prob['problem']}\n</problem>\n<answer>"
                
                # In production, call model.generate()
                # For now, placeholder
                response = "[placeholder]"
                
                # Check if answer is in response
                if prob['answer'].lower() in response.lower():
                    correct += 1
            
            results[category] = {
                "correct": correct,
                "total": total,
                "accuracy": correct / total * 100 if total > 0 else 0
            }
        
        # Overall
        total_correct = sum(r["correct"] for r in results.values())
        total_problems = sum(r["total"] for r in results.values())
        results["overall"] = {
            "correct": total_correct,
            "total": total_problems,
            "accuracy": total_correct / total_problems * 100 if total_problems > 0 else 0
        }
        
        self.results = results
        return results
    
    def save_benchmark(self, output_path: str):
        """Salva benchmark su file"""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump({
                "benchmarks": self.benchmarks,
                "results": self.results
            }, f, indent=2)
        
        print(f"Benchmark saved to {output_path}")
    
    def print_report(self):
        """Stampa report risultati"""
        
        print("\n" + "=" * 60)
        print("ODIN-100M Reasoning Benchmark Report")
        print("=" * 60)
        
        for category, metrics in self.results.items():
            if category != "overall":
                print(f"\n📊 {category.upper()}")
                print(f"   Accuracy: {metrics['accuracy']:.1f}%")
                print(f"   Correct: {metrics['correct']}/{metrics['total']}")
        
        print("\n" + "-" * 60)
        overall = self.results.get("overall", {})
        print(f"🎯 OVERALL ACCURACY: {overall.get('accuracy', 0):.1f}%")
        print("=" * 60)


# Targets
TARGETS = {
    "arithmetic": 80.0,  # Should be high for synthetic-trained model
    "algebra": 70.0,
    "logic": 60.0,
    "code": 50.0,
    "overall": 65.0
}


def run_benchmark():
    """Esegue benchmark completo"""
    
    print("Initializing Reasoning Benchmark...")
    
    bench = ReasoningBenchmark()
    
    # Show benchmark stats
    print("\nBenchmark Suite:")
    for cat, probs in bench.benchmarks.items():
        print(f"  • {cat}: {len(probs)} problems")
    
    # Save benchmark
    bench.save_benchmark("benchmarks/reasoning_benchmark.json")
    
    # Targets
    print("\n🎯 Target Accuracies:")
    for cat, target in TARGETS.items():
        print(f"  • {cat}: {target}%")
    
    print("\n[Waiting for trained model to run evaluation]")


if __name__ == "__main__":
    run_benchmark()
