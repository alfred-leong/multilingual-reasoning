import json
import os

def fix_jsonl(file_path):
    updated_lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            # 1. Find entries where english_ans=translated_ans is false
            # and change translated_answer to english_answer
            if not entry.get("english_ans=translated_ans", True):
                entry["translated_answer"] = entry["english_answer"]
            
            # 2. Change english_ans=translated_ans to true for all entries
            entry["english_ans=translated_ans"] = True
            
            updated_lines.append(json.dumps(entry, ensure_ascii=False))
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in updated_lines:
            f.write(line + '\n')

if __name__ == "__main__":
    for filename in os.listdir("."):
        if filename.endswith(".jsonl"):
            print(f"Processing {filename}...")
            fix_jsonl(filename)
            print(f"Finished {filename}.")
    print("All files processed.")
