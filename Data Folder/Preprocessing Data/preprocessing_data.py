## preprocessing_data.py
## 데이터 전처리 코드
  
import re
import json
import pandas as pd
import random

## 전처리 함수
def preprocess_text(text):
    text = re.sub(r'\.{3,}', '...', text.replace('…', '.'))
    text = text.replace('─', '').replace('―', '').replace('-', '')
    text = text.replace('『', '').replace('』', '')
    text = text.replace('【', '"').replace('】', '"')
    text = text.replace('「', '"').replace('」', '"')
    text = text.replace('ㆍ', ' ').replace('(爆裂道)', '').replace('ㄸ', '또')
    return text

## 텍스트를 분할하고 문장 순서를 섞는 함수
def split_and_shuffle_sentences_with_offset(text):
    pattern = r'([.]{3}|[?!]{2}|[.?!])'
    sentences = re.split(pattern, text)
    sentences = [''.join(x) for x in zip(sentences[0::2], sentences[1::2])]
    preserved_sentences = sentences[:6]
    shuffled_sentences = []
    temp_sentence = []
    for sentence in sentences[6:]:
        if '...' in sentence or '?!' in sentence:
            if temp_sentence:
                random.shuffle(temp_sentence)
                shuffled_sentences.extend(temp_sentence)
                temp_sentence = []
            shuffled_sentences.append(sentence)
        else:
            temp_sentence.append(sentence)
    if temp_sentence:
        random.shuffle(temp_sentence)
        shuffled_sentences.extend(temp_sentence)
    return ' '.join(preserved_sentences + shuffled_sentences)

## 데이터 로드 함수
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))

## 데이터 전처리 및 증강 함수
def preprocess_and_augment(input_file1, input_file2, output_file):
    df1 = load_data(input_file1)
    df1['prompt'] = df1['prompt'].apply(preprocess_text)
    df1['response'] = df1['response'].apply(preprocess_text)
    df1 = df1[df1['response'].apply(len) >= 3]

    df2 = load_data(input_file2).iloc[:940]
    df2.columns = ['prompt', 'response']
    df2['prompt'] = df2['prompt'].apply(lambda x: f"사용자 : {x}")
    df2['response'] = df2['response'].apply(lambda x: f"메구밍 : {x}")

    combined_df = pd.concat([df1[['prompt', 'response']], df2], axis=0)

    augmented_responses = combined_df['response'].apply(split_and_shuffle_sentences_with_offset)
    augmented_df = pd.concat([combined_df, pd.DataFrame({'prompt': combined_df['prompt'], 'response': augmented_responses})], ignore_index=True)

    augmented_df = augmented_df.drop_duplicates(subset=['prompt', 'response']).reset_index(drop=True)
    json_data = augmented_df[['prompt', 'response']].to_dict(orient='records')

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    print(f"데이터 처리 및 저장 완료: {output_file}")

if __name__ == "__main__":
    preprocess_and_augment('pre_dialogues.json', 'dialogues.json', 'augmented_prompt_response.json')
