import json
import glob
import os

files = glob.glob('data/translate_dpo/translated/*.jsonl')

for file in files:
    total_correct_english = 0
    total_correct_native = 0
    total_same_answers = 0
    total_diff_answers = 0
    total_eng_correct_native_same = 0
    total_eng_correct_native_diff = 0
    total_number_of_questions = 0

    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            total_number_of_questions += 1
            english_answer = data.get('english_answer')
            native_answer = data.get('native_answer')
            gold_answer = data.get('gold_answer')
            
            if english_answer == gold_answer:
                total_correct_english += 1
                if english_answer == native_answer:
                    total_eng_correct_native_same += 1
                else:
                    total_eng_correct_native_diff += 1
            
            if native_answer == gold_answer:
                total_correct_native += 1
                
            if english_answer == native_answer:
                total_same_answers += 1
            else:
                total_diff_answers += 1

    print(f"File: {os.path.basename(file)}, Total number of questions: {total_number_of_questions}")
    print(f"1. Number of correct english_answers: {total_correct_english} ({total_correct_english/total_number_of_questions*100:.2f}%)")
    print(f"2. Number of correct native answers: {total_correct_native} ({total_correct_native/total_number_of_questions*100:.2f}%)")
    print(f"3. Number of same english_answers and native_answers: {total_same_answers} ({total_same_answers/total_number_of_questions*100:.2f}%)")
    print(f"4. Number of different english_answers and native_answers: {total_diff_answers} ({total_diff_answers/total_number_of_questions*100:.2f}%)")
    print(f"5. Number of correct english answers, and same native answers: {total_eng_correct_native_same} ({total_eng_correct_native_same/total_number_of_questions*100:.2f}%)")
    print(f"6. Number of correct english answers, and different native answers: {total_eng_correct_native_diff} ({total_eng_correct_native_diff/total_number_of_questions*100:.2f}%)")
    print("-" * 40)
