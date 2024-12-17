## training_translation_model.py
## 번역 모델을 메구밍처럼 번역하도록 학습시키는 코드

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from huggingface_hub import HfApi, HfFolder
from datasets import Dataset
import pandas as pd
import torch
import os

hf_token = "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"
os.environ["HUGGINGFACE_TOKEN"] = hf_token
HfFolder.save_token(hf_token)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(device)

df = pd.read_csv('data_for_training_translation_model.csv')
dataset = Dataset.from_pandas(df)

# 학습 및 검증 데이터셋 분리 (80% 학습, 20% 검증)
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

def preprocess_function(examples):
    # 영어 텍스트와 한국어 텍스트를 토큰화
    inputs = tokenizer(examples["response_en"], padding="max_length", truncation=True, max_length=1024)
    targets = tokenizer(examples["response_ko"], padding="max_length", truncation=True, max_length=1024)
    inputs["labels"] = targets["input_ids"]  # 레이블 설정
    return inputs

# 전처리 함수 적용
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# 훈련 인자 설정 (GPU 환경에 맞게)
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True  # GPU에서 더 빠르게 학습하도록 FP16 사용
)

# Trainer 설정
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer
)

# 모델 학습
trainer.train()

# 모델 저장
model.save_pretrained("./finetuned_nllb_600M")
tokenizer.save_pretrained("./finetuned_nllb_600M")
