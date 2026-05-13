import json
import pandas as pd
from pathlib import Path
from sys import argv

def build_bon_dataframe(json_directory, output_path):
    all_bags = []
    base_dir = Path(json_directory)
    
    # 1. Recursive search using rglob
    # Finds any file matching the pattern in any subdirectory
    file_paths = list(base_dir.rglob("*.bag-of-nodes.json"))
    
    if not file_paths:
        print(f"No files matching '*.bag-of-nodes.json' found in {json_directory}")
        return

    for path in file_paths:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                
                # 2. Use the relative path as the ID
                # Preserves folder structure (e.g., student names) in the data
                data['file_id'] = str(path.relative_to(base_dir))
                all_bags.append(data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading {path}: {e}")

    # 3. Create the master feature matrix
    # Pandas automatically aligns columns and we fill missing values with 0
    df = pd.DataFrame(all_bags).fillna(0)
    
    # Reorder to ensure file_id is the first column
    cols = ['file_id'] + [c for c in df.columns if c != 'file_id']
    df = df[cols]
    
    # 4. Save to CSV
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(out_file, index=False)
    print(f"Adapter finished: {len(all_bags)} files indexed into {output_path}")

if __name__ == "__main__":
    build_bon_dataframe(argv[1], argv[1] +'processed/bags_of_nodes.csv')