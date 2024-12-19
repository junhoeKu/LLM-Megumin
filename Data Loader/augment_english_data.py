## augment_english_data.py
## 영어 데이터 증강하는 코드, 1) 임의의 부사 삽입 2) 문장 내 두 단어를 임의로 교체

import json
import random
import copy

## 부사 리스트
adverbs = [
    "quickly", "carefully", "suddenly", "quietly", "bravely", "cheerfully", "eagerly", "secretly",
    "lazily", "gracefully", "hastily", "frequently", "gently", "boldly", "dramatically", "softly",
    "thoughtfully", "calmly", "wildly", "awkwardly", "carelessly", "nervously", "instantly", "reluctantly",
    "deliberately", "happily", "sadly", "enthusiastically", "noisily", "politely", "slowly", "brightly",
    "patiently", "reluctantly", "recklessly", "regularly", "sternly", "vigorously", "elegantly", "steadily",
    "kindly", "frantically", "courageously", "respectfully", "unexpectedly", "powerfully", "smoothly",
    "indifferently", "passionately", "firmly"
]

## 함수: 임의의 부사를 문장에 삽입
def insert_random_adverb(text):
    if text is None:
        return text
    words = text.split()
    if not words:
        return text
    index = random.randint(0, len(words) - 1)
    adverb = random.choice(adverbs)
    words.insert(index, adverb)
    return ' '.join(words)

## 함수: 문장 내 두 단어를 임의로 교체
def swap_two_words(text):
    if text is None:
        return text
    words = text.split()
    if len(words) < 2:
        return text
    idx1, idx2 = random.sample(range(len(words)), 2)
    words[idx1], words[idx2] = words[idx2], words[idx1]
    return ' '.join(words)

## 데이터 증강 함수
def augment_data(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    augmented_data = []

    for entry in data:
        ## 1. 원본 데이터 추가
        augmented_data.append(entry)

        ## 2. 부사 삽입 버전 추가
        adverb_version = copy.deepcopy(entry)
        adverb_version["context"] = insert_random_adverb(adverb_version["context"])
        adverb_version["response"] = insert_random_adverb(adverb_version["response"])
        augmented_data.append(adverb_version)

        ## 3. 단어 위치 변경 버전 추가
        swap_version = copy.deepcopy(entry)
        swap_version["context"] = swap_two_words(swap_version["context"])
        swap_version["response"] = swap_two_words(swap_version["response"])
        augmented_data.append(swap_version)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(augmented_data, f, ensure_ascii=False, indent=4)

    print(f"데이터 증강 완료! 총 {len(augmented_data)}개의 항목이 있습니다.")

if __name__ == "__main__":
    INPUT_FILE = "../LLM/Preprocessing/Lucy5_English_1211.json"
    OUTPUT_FILE = "augmented_Lucy5_English_1211.json"
    augment_data(INPUT_FILE, OUTPUT_FILE)