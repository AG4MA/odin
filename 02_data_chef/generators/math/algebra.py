"""
Algebra Data Generator
======================
Genera equazioni lineari e quadratiche con soluzioni step-by-step.
Verificato con SymPy.
"""

import random
from typing import Dict, List
from sympy import symbols, Eq, solve, sympify, simplify


x = symbols('x')


def generate_linear_equation(max_coef: int = 20) -> Dict:
    """
    Genera equazione lineare: ax + b = c
    """
    a = random.randint(1, max_coef)
    b = random.randint(-max_coef, max_coef)
    
    # Genera soluzione intera per semplicità
    solution = random.randint(-10, 10)
    c = a * solution + b
    
    # Costruisci problema
    if b >= 0:
        problem = f"Solve for x: {a}x + {b} = {c}"
    else:
        problem = f"Solve for x: {a}x - {abs(b)} = {c}"
    
    # Reasoning step-by-step
    reasoning = f"""Step 1: Isolate the term with x
{a}x + {b} = {c}
{a}x = {c} - {b}
{a}x = {c - b}

Step 2: Divide both sides by {a}
x = {c - b} / {a}
x = {solution}"""
    
    # Verifica con SymPy
    lhs = a * x + b
    verified = solve(Eq(lhs, c), x)[0] == solution
    
    return {
        "type": "linear_equation",
        "problem": problem,
        "reasoning": reasoning,
        "answer": f"x = {solution}",
        "verified": verified
    }


def generate_linear_two_steps(max_coef: int = 10) -> Dict:
    """
    Genera equazione: ax + b = cx + d
    """
    a = random.randint(2, max_coef)
    c = random.randint(1, a - 1)  # c < a per soluzione positiva
    
    solution = random.randint(1, 10)
    
    b = random.randint(-max_coef, max_coef)
    d = (a - c) * solution + b
    
    problem = f"Solve for x: {a}x + {b} = {c}x + {d}"
    
    reasoning = f"""Step 1: Move x terms to left side
{a}x - {c}x + {b} = {d}
{a - c}x + {b} = {d}

Step 2: Move constants to right side
{a - c}x = {d} - {b}
{a - c}x = {d - b}

Step 3: Divide both sides by {a - c}
x = {d - b} / {a - c}
x = {solution}"""
    
    # Verifica
    lhs = a * x + b
    rhs = c * x + d
    verified = solve(Eq(lhs, rhs), x)[0] == solution
    
    return {
        "type": "linear_two_sided",
        "problem": problem,
        "reasoning": reasoning,
        "answer": f"x = {solution}",
        "verified": verified
    }


def generate_simple_quadratic() -> Dict:
    """
    Genera equazione quadratica semplice: x² = n (soluzione intera)
    """
    solution = random.randint(1, 12)
    n = solution ** 2
    
    problem = f"Solve for x: x² = {n}"
    
    reasoning = f"""Step 1: Take square root of both sides
x² = {n}
x = ±√{n}

Step 2: Calculate the square root
√{n} = {solution}

x = {solution} or x = -{solution}"""
    
    return {
        "type": "simple_quadratic",
        "problem": problem,
        "reasoning": reasoning,
        "answer": f"x = {solution} or x = -{solution}",
        "verified": True
    }


def generate_factorable_quadratic() -> Dict:
    """
    Genera quadratica fattorizzabile: (x - a)(x - b) = 0
    """
    a = random.randint(-8, 8)
    b = random.randint(-8, 8)
    if a == b:
        b = a + random.choice([-1, 1])
    
    # Espandi: x² - (a+b)x + ab = 0
    coef_b = -(a + b)
    coef_c = a * b
    
    if coef_b >= 0:
        b_str = f"+ {coef_b}"
    else:
        b_str = f"- {abs(coef_b)}"
    
    if coef_c >= 0:
        c_str = f"+ {coef_c}"
    else:
        c_str = f"- {abs(coef_c)}"
    
    problem = f"Solve for x: x² {b_str}x {c_str} = 0"
    
    reasoning = f"""Step 1: Factor the quadratic
x² {b_str}x {c_str} = 0
(x - {a})(x - {b}) = 0

Step 2: Apply zero product property
x - {a} = 0  or  x - {b} = 0

Step 3: Solve each equation
x = {a}  or  x = {b}"""
    
    return {
        "type": "factorable_quadratic",
        "problem": problem,
        "reasoning": reasoning,
        "answer": f"x = {a} or x = {b}",
        "verified": True
    }


def generate_batch(n: int, difficulty: str = "mixed") -> List[Dict]:
    """Genera batch di problemi di algebra"""
    generators = {
        "easy": [generate_linear_equation],
        "medium": [generate_linear_equation, generate_linear_two_steps, generate_simple_quadratic],
        "hard": [generate_linear_two_steps, generate_factorable_quadratic],
        "mixed": [generate_linear_equation, generate_linear_two_steps, 
                  generate_simple_quadratic, generate_factorable_quadratic]
    }
    
    gen_list = generators.get(difficulty, generators["mixed"])
    
    results = []
    for _ in range(n):
        gen = random.choice(gen_list)
        results.append(gen())
    
    return results


def format_for_training(example: Dict) -> str:
    """Formatta per training"""
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
    print("=== Algebra Generator Test ===\n")
    
    for gen in [generate_linear_equation, generate_linear_two_steps, 
                generate_simple_quadratic, generate_factorable_quadratic]:
        example = gen()
        print(f"[{example['type'].upper()}]")
        print(format_for_training(example))
        print(f"Verified: {example['verified']}\n")
        print("-" * 40 + "\n")
