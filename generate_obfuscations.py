import sys
import os
import ollama

# Define the obfuscation prompts corresponding to levels L1-L6
obfuscation_prompts = {
    "L1": "You are a code obfuscator. Modify the provided code by applying L1 plagiarism: change only whitespace, formatting, and add/remove comments. Do not change any logic or identifiers. Output only the raw code.",
    "L2": "You are a code obfuscator. Modify the provided code by applying L2 plagiarism: rename identifiers (variables, functions, classes). You may also include L1 whitespace/comment changes. Keep the program logic identical. Output only the raw code.",
    "L3": "You are a code obfuscator. Modify the provided code by applying L3 plagiarism: change the location, order, and visibility of declarations, and introduce dummy variables. Include L1 and L2 changes. Output only the raw code.",
    "L4": "You are a code obfuscator. Modify the provided code by applying L4 plagiarism: perform method extractions and declare dummy methods. Include L1 to L3 changes. Output only the raw code.",
    "L5": "You are a code obfuscator. Modify the provided code by applying L5 plagiarism: alter method bodies by changing API calls, data types, operators, control structures, and the order of operations. Include L1 to L4 changes. Output only the raw code.",
    "L6": "You are a code obfuscator. Modify the provided code by applying L6 plagiarism: make fundamental changes to the decision logic. Introduce entirely new control structures, shift loop boundaries, or switch between iterative (loops) and recursive code. Include L1 to L5 changes. Output only the raw code."
}

def clean_output(text):
    """Strip markdown code blocks from the LLM output if they exist."""
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()

def process_file(filepath, model_name="granite4.1:8b"):
    if not os.path.isfile(filepath):
        print(f"File not found: {filepath}")
        return

    print(f"\n--- Processing {filepath} ---")
    with open(filepath, "r", encoding="utf-8") as f:
        original_code = f.read()

    # Split the filename so we can insert the -L# suffix before the extension
    base_name, ext = os.path.splitext(filepath)

    for level, instruction in obfuscation_prompts.items():
        output_file = f"{base_name}-{level}{ext}"
        print(f"  Generating {level} obfuscation -> {output_file}")
        
        prompt = f"{instruction}\n\nHere is the code to obfuscate:\n\n```\n{original_code}\n```"
        
        try:
            # Call the local Ollama model
            response = ollama.generate(
                model=model_name,
                prompt=prompt,
                options={
                    "temperature": 0.7
                }
            )
            
            generated_code = clean_output(response['response'])
            
            with open(output_file, "w", encoding="utf-8") as out_f:
                out_f.write(generated_code)
                
        except Exception as e:
            print(f"  Error generating {level} for {filepath}: {e}")

if __name__ == "__main__":
    # Ensure at least one file is passed as an argument
    if len(sys.argv) < 2:
        print("Usage: python generate_obfuscations.py <file1> <file2> ...")
        sys.exit(1)

    # sys.argv[1:] contains the list of files passed through the CLI
    for file_arg in sys.argv[1:]:
        process_file(file_arg, model_name="granite4.1:8b")
