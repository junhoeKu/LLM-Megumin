## honorific_classify.py
## response column이 존댓말인지 여부를 평가하는 코드

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

## 모델 초기화 함수
def initialize_model(model_name):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device

## 존댓말 판단 함수
def classify_honorific(sentence, tokenizer, model, device):
    if not isinstance(sentence, str):
        return 0
    inputs = tokenizer(sentence.split('.')[0], return_tensors="pt", padding=True, truncation=True, max_length=300).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.argmax(outputs.logits, dim=-1).item()

## 데이터 처리 함수
def process_responses(input_file, output_file, model_name):
    tokenizer, model, device = initialize_model(model_name)
    df = pd.read_excel(input_file)
    df['is_honorific'] = df['response'].apply(lambda x: classify_honorific(x, tokenizer, model, device))
    df.to_excel(output_file, index=False)
    print(f"존댓말 컬럼이 생성 완료되었습니다. 저장 경로: {output_file}")

if __name__ == "__main__":
    MODEL_NAME = "beomi/kcbert-base"
    INPUT_FILE = "responses.xlsx"
    OUTPUT_FILE = "responses_with_honorifics.xlsx"
    process_responses(INPUT_FILE, OUTPUT_FILE, MODEL_NAME)
