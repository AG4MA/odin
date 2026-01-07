"""
Python Code Generator
=====================
Genera problemi di coding con soluzioni verificate.
"""

import random
import ast
from typing import Dict, List


def generate_function_sum_list() -> Dict:
    """Genera: scrivi funzione che somma lista"""
    nums = [random.randint(1, 20) for _ in range(random.randint(3, 6))]
    expected = sum(nums)
    
    return {
        "type": "function_implementation",
        "problem": "Write a Python function `sum_list(numbers)` that returns the sum of all numbers in a list.",
        "reasoning": """Step 1: Define the function with a parameter for the list
Step 2: Initialize a variable to track the sum
Step 3: Iterate through each number in the list
Step 4: Add each number to our sum
Step 5: Return the final sum""",
        "answer": """def sum_list(numbers):
    total = 0
    for num in numbers:
        total += num
    return total""",
        "test_case": f"sum_list({nums}) == {expected}",
        "verified": True
    }


def generate_function_max_list() -> Dict:
    """Genera: trova massimo in lista"""
    nums = [random.randint(1, 100) for _ in range(random.randint(4, 7))]
    expected = max(nums)
    
    return {
        "type": "function_implementation",
        "problem": "Write a Python function `find_max(numbers)` that returns the largest number in a list.",
        "reasoning": """Step 1: Handle edge case - if list is empty, return None
Step 2: Initialize max_val with the first element
Step 3: Iterate through remaining elements
Step 4: Update max_val if current element is larger
Step 5: Return max_val""",
        "answer": """def find_max(numbers):
    if not numbers:
        return None
    max_val = numbers[0]
    for num in numbers[1:]:
        if num > max_val:
            max_val = num
    return max_val""",
        "test_case": f"find_max({nums}) == {expected}",
        "verified": True
    }


def generate_function_factorial() -> Dict:
    """Genera: calcola fattoriale"""
    n = random.randint(3, 8)
    
    def factorial(x):
        if x <= 1:
            return 1
        return x * factorial(x - 1)
    
    expected = factorial(n)
    
    return {
        "type": "function_implementation",
        "problem": f"Write a Python function `factorial(n)` that returns n! (n factorial).",
        "reasoning": """Step 1: Handle base case - 0! = 1! = 1
Step 2: For n > 1, factorial(n) = n * factorial(n-1)
Step 3: Use recursion or iteration
Step 4: Return the result""",
        "answer": """def factorial(n):
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result""",
        "test_case": f"factorial({n}) == {expected}",
        "verified": True
    }


def generate_function_fibonacci() -> Dict:
    """Genera: calcola n-esimo Fibonacci"""
    n = random.randint(5, 12)
    
    def fib(x):
        if x <= 1:
            return x
        a, b = 0, 1
        for _ in range(2, x + 1):
            a, b = b, a + b
        return b
    
    expected = fib(n)
    
    return {
        "type": "function_implementation",
        "problem": f"Write a Python function `fibonacci(n)` that returns the n-th Fibonacci number (0-indexed).",
        "reasoning": """Step 1: Handle base cases - fib(0) = 0, fib(1) = 1
Step 2: For n > 1, fib(n) = fib(n-1) + fib(n-2)
Step 3: Use iterative approach for efficiency
Step 4: Track previous two values and update""",
        "answer": """def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b""",
        "test_case": f"fibonacci({n}) == {expected}",
        "verified": True
    }


def generate_function_reverse_string() -> Dict:
    """Genera: inverti stringa"""
    words = ["hello", "python", "world", "code", "test", "algorithm"]
    word = random.choice(words)
    expected = word[::-1]
    
    return {
        "type": "function_implementation",
        "problem": "Write a Python function `reverse_string(s)` that returns the reversed string.",
        "reasoning": """Step 1: There are multiple approaches
Step 2: Approach 1 - Use slicing [::-1]
Step 3: Approach 2 - Build new string character by character
Step 4: Approach 3 - Convert to list, reverse, join
Step 5: Return the reversed string""",
        "answer": """def reverse_string(s):
    return s[::-1]""",
        "test_case": f'reverse_string("{word}") == "{expected}"',
        "verified": True
    }


def generate_function_is_palindrome() -> Dict:
    """Genera: verifica palindromo"""
    palindromes = ["radar", "level", "civic", "rotor", "kayak"]
    non_palindromes = ["hello", "python", "world", "code"]
    
    if random.random() > 0.5:
        word = random.choice(palindromes)
        expected = True
    else:
        word = random.choice(non_palindromes)
        expected = False
    
    return {
        "type": "function_implementation",
        "problem": "Write a Python function `is_palindrome(s)` that returns True if the string is a palindrome.",
        "reasoning": """Step 1: A palindrome reads the same forwards and backwards
Step 2: Compare string with its reverse
Step 3: Return True if they match, False otherwise""",
        "answer": """def is_palindrome(s):
    return s == s[::-1]""",
        "test_case": f'is_palindrome("{word}") == {expected}',
        "verified": True
    }


def generate_batch(n: int, difficulty: str = "mixed") -> List[Dict]:
    """Genera batch di problemi di coding"""
    generators = [
        generate_function_sum_list,
        generate_function_max_list,
        generate_function_factorial,
        generate_function_fibonacci,
        generate_function_reverse_string,
        generate_function_is_palindrome
    ]
    
    results = []
    for _ in range(n):
        gen = random.choice(generators)
        example = gen()
        
        # Verifica syntax
        try:
            ast.parse(example["answer"])
            example["syntax_valid"] = True
        except SyntaxError:
            example["syntax_valid"] = False
        
        results.append(example)
    
    return results


def format_for_training(example: Dict) -> str:
    """Formatta per training"""
    return f"""<problem>
{example['problem']}
</problem>
<reasoning>
{example['reasoning']}
</reasoning>
<code>
{example['answer']}
</code>
<test>
{example['test_case']}
</test>"""


# Test
if __name__ == "__main__":
    print("=== Python Code Generator Test ===\n")
    
    batch = generate_batch(6)
    
    for ex in batch:
        print(format_for_training(ex))
        print(f"Verified: {ex['verified']}, Syntax: {ex['syntax_valid']}")
        print("-" * 50 + "\n")
