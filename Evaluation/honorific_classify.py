## honorific_classify.py
## response column이 존댓말인지 여부를 평가하는 코드

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

## 1. 모델과 토크나이저 로드
MODEL_NAME = "beomi/kcbert-base"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

## 2. 데이터 로드
df = pd.read_excel("responses.xlsx")

## 3. 존댓말 판단 함수
def classify_honorific(sentence):
    sentence = sentence.split('.')[0] if isinstance(sentence, str) else ""
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=300).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()  ## 예측 결과 (0 or 1)
    return prediction

## 4. 존댓말 분류 후 새로운 컬럼 생성
df['is_honorific'] = df['response'].apply(lambda x: classify_honorific(x) if isinstance(x, str) else 0)

## 5. 결과 저장
df.to_excel("responses.xlsx", index=False)
print('존댓말 컬럼이 생성 완료 되었습니다.')
