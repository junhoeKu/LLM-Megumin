import torch
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForLanguageModeling
from data_loader import load_megumin_dialogues_to_blessing, load_megumin_dialogues_to_flame, load_megumin_qa_dialogues

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 토크나이저와 모델 로드 및 GPU 이동
tokenizer = AutoTokenizer.from_pretrained("lcw99/t5-base-korean-chit-chat")
model = AutoModelForSeq2SeqLM.from_pretrained("lcw99/t5-base-korean-chit-chat").to(device)

# 대사 데이터를 토크나이징
def tokenize_dialogues(dialogues):
    dialogue_dict = {
        "context": [dialogue["context"] for dialogue in dialogues],  # 문맥
        "response": [dialogue["response"] for dialogue in dialogues]  # 응답
    }

    dataset = Dataset.from_dict(dialogue_dict)

    tokenizer.pad_token = tokenizer.eos_token

    # 데이터셋 토큰화
    tokenized_dataset = dataset.map(
        lambda x: {
            'input_ids': tokenizer(
                "[일반] " + x["context"],
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )["input_ids"].squeeze(0),  # 첫 번째 차원 제거
            
            'labels': tokenizer(
                "[메구밍] " + x["response"],
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )["input_ids"].squeeze(0)
        }
    )

    return tokenized_dataset

# 데이터를 train/validation으로 나누기 (10%를 validation으로)
def split_train_validation(tokenized_dataset):
    train_test_split_result = tokenized_dataset.train_test_split(test_size=0.1, shuffle=True)
    return train_test_split_result['train'], train_test_split_result['test']

# Fine Tuning 설정 및 학습 진행
def fine_tune_model(tokenized_dataset):
    train_dataset, val_dataset = split_train_validation(tokenized_dataset)

    # 학습 설정
    training_args = Seq2SeqTrainingArguments(
        output_dir="./megumin_results",     # 모델이 저장될 경로
        learning_rate=2e-5,                 # 초기 학습률 설정
        lr_scheduler_type="cosine",         # 스케줄러 적용
        warmup_steps=500,                   # Warmup 스텝
        overwrite_output_dir=True,          # 경로 덮어쓰기
        num_train_epochs=5,                 # 학습할 epoch 수
        per_device_train_batch_size=4,      # 배치 크기
        gradient_accumulation_steps=8,      # 8 배치 누적 후 그래디언트 업데이트
        save_steps=500,                     # 저장 주기
        save_total_limit=1,                 # 저장할 모델 체크포인트 수
        logging_dir="./megumin_logs",       # 로그 저장 디렉토리
        logging_steps=100,                  # 로그 기록 주기
        logging_first_step=False,           # 첫 스텝 로깅 비활성화
        evaluation_strategy="steps",        # 검증 설정 (훈련 중 일정 주기로 검증)
        eval_steps=500,                     # 검증 주기 (매 500 스텝마다)
        load_best_model_at_end=True,        # 가장 좋은 모델 저장
        fp16=True,                          # float16 활성화
    )

    # Trainer 설정
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,          # 검증 데이터 추가
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8),
    )

    # 모델 Fine Tuning 진행
    trainer.train()

    # 모델 저장
    model.save_pretrained("./model_t5_chat_5epoch_8batch")
    tokenizer.save_pretrained("./tokenizer_t5_chat_5epoch_8batch")

if __name__ == "__main__":
    ## 1. JSON 파일에서 메구밍 대사 불러오기
    with open("Json_Data/aug_output.json", "r", encoding="utf-8") as f:
        dialogues = json.load(f)

    tokenized_megumin_dataset = tokenize_dialogues(dialogues)

    print(tokenized_megumin_dataset[0])