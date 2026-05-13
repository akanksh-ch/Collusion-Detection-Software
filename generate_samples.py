import os
import requests
from pathlib import Path

# =========================
# CONFIG
# =========================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "granite4.1:8b"

OUTPUT_DIR = "fibonacci_c_solutions"

METHODS = {
    "recursive": {
        "description": "plain recursive Fibonacci in C",
        "count": 5,
    },
    "iterative": {
        "description": "iterative Fibonacci in C using loops",
        "count": 5,
    },
    "memoization": {
        "description": "memoized recursive Fibonacci in C",
        "count": 5,
    },
    "binet": {
        "description": "Binet formula Fibonacci in C",
        "count": 5,
    },
    "matrix_exp": {
        "description": "matrix exponentiation Fibonacci in C",
        "count": 5,
    },
}

# =========================
# PROMPT TEMPLATE
# =========================

PROMPT_TEMPLATE = """
Generate a COMPLETE standalone C program.

Requirements:
- Implement Fibonacci using the following method:
  {method_description}

- Make this implementation DIFFERENT from previous variants.
- Use different:
  - variable names
  - structure
  - formatting
  - helper functions
  - comments
  - input/output style

- The code must:
  - compile with gcc
  - include main()
  - ask user for n
  - print Fibonacci result
  - be valid pure C

- Return ONLY raw C code.
- Do not use markdown.
- Do not use triple backticks.
"""

# =========================
# OLLAMA CALL
# =========================

def generate_code(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    data = response.json()
    return data["response"]


# =========================
# CLEAN RESPONSE
# =========================

def clean_code(code: str) -> str:
    code = code.strip()

    # Remove markdown fences if model ignores instruction
    if code.startswith("```"):
        lines = code.splitlines()

        # remove first ```
        lines = lines[1:]

        # remove last ```
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]

        code = "\n".join(lines)

    return code.strip()


# =========================
# SAVE FILE
# =========================

def save_code(method_name: str, index: int, code: str):
    method_dir = Path(OUTPUT_DIR) / method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    filename = method_dir / f"fib_{method_name}_{index + 1}.c"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(code)

    print(f"[+] Saved: {filename}")


# =========================
# MAIN
# =========================

def main():
    for method_name, config in METHODS.items():
        print(f"\n=== Generating {method_name} variants ===")

        for i in range(config["count"]):
            print(f"  -> Variant {i + 1}")

            prompt = PROMPT_TEMPLATE.format(
                method_description=config["description"]
            )

            try:
                code = generate_code(prompt)
                code = clean_code(code)

                save_code(method_name, i, code)

            except Exception as e:
                print(f"[!] Failed variant {i + 1}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()