## training.py

import torch
import random
from torch.cuda.amp import autocast
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig, EarlyStoppingCallback
from peft import get_peft_model, LoraConfig, TaskType, PrefixTuningConfig
from data_loader import load_megumin_dialogues_to_blessing, load_megumin_dialogues_to_flame, load_megumin_qa_dialogues

## Early stopping callback 정의
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=10)  ## 성능이 10번 연속 개선되지 않으면 학습 중단

## Data Collator
class DataCollatorWithProfile(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, profile_text, prob_profile=0.25, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.profile_text = profile_text
        self.prob_profile = prob_profile
        self.tokenizer = tokenizer

    def __call__(self, features):
        for feature in features:
            # 25% 확률로 프로필을 추가
            if random.random() < self.prob_profile:
                # feature['input_ids']가 리스트인지 텐서인지 확인
                if isinstance(feature['input_ids'], list):
                    decoded_text = self.tokenizer.decode(feature['input_ids'])
                else:
                    decoded_text = self.tokenizer.decode(feature['input_ids'].tolist())
                
                # 프로필 텍스트와 결합하고 다시 토큰화
                combined_text = self.profile_text + " " + decoded_text
                tokenized_input = self.tokenizer(
                    combined_text,
                    max_length=1024,
                    truncation=True,
                    padding='max_length',
                    return_tensors="pt"
                )["input_ids"]

                # 텐서를 리스트로 변환하여 처리
                feature['input_ids'] = tokenized_input.squeeze(0).tolist()  # 텐서를 리스트로 변환
        return super().__call__(features)

## 메구밍 프로필 정의
megumin_profile_metadata = (
    "[Megumin Profile]\n"
    "Name: Megumin\n"
    "Age: 12 (Crimson Demon Clan)\n"
    "Role: Arch Wizard\n"
    "Abilities: Can only use Explosion Magic; possesses strong wizard stats\n"
    "Personality: Proud, chuunibyou (adolescent delusions of grandeur), tsundere traits, cares for her companions, loves cute things\n"
    "Appearance: Black hair down to her shoulders, red eyes, wears a cute red dress\n"
    "Relations: Kazuma (companion, respect, love), Aqua (companion), Darkness (companion)\n"
    "Family: Father Hyoizaburo, Mother Yuiyui, younger sister Komekko\n"
    "Traits: Rejects any magic other than Explosion Magic, introduces herself with chuunibyou flair\n"
    "Famous Quote: 'My name is Megumin! The greatest wizard of the Crimson Demon Clan, wielder of Explosion Magic! EXPLOSION!!'\n"
)

## 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## 토크나이저와 모델 로드 및 GPU 이동
## Qwen/Qwen2.5-7B-Instruct
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

## 4-bit 양자화를 위한 BitsAndBytesConfig 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      ## 4-bit 양자화 적용
    bnb_4bit_use_double_quant=True,         ## double quantization 적용
    bnb_4bit_quant_type="nf4",              ## Quantization type (nf4가 기본), 메모리 부족할땐 fp4
    bnb_4bit_compute_dtype=torch.bfloat16   ## 연산에 bf16 사용
)

## 4-bit 양자화된 모델 로드
## Qwen/Qwen2.5-7B-Instruct
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", 
    quantization_config=bnb_config, 
    device_map="auto"  ## 자동으로 GPU에 로드
)

## LoRA 설정
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,           ## Task type을 설정 (언어 모델링)
    inference_mode=False,                   ## 학습 모드로 설정
    r=32,                                   ## low-rank adaptation 차원
    lora_alpha=64,                          ## LoRA scaling factor
    lora_dropout=0.2,                       ## 드롭아웃 적용 (Optional)
    ## LoRA가 적용될 레이어를 명시적으로 지정
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "ff_proj"]  # 더 많은 레이어에 LoRA 적용
)

## 모델에 QLoRA 적용
model = get_peft_model(model, lora_config)

## Gradient Checkpointing 적용
model.gradient_checkpointing_enable()   ## Gradient Checkpointing 활성화

def tokenize_dialogues(dialogues, max_length=1024):
    dialogue_dict = {
        "combined": [f"scene: {dialogue['scene_context']} context: {dialogue['context']} megumin: {dialogue['response']}" for dialogue in dialogues]
    }

    ## Dataset 객체로 변환
    dataset = Dataset.from_dict(dialogue_dict)

    ## Padding 토큰 정의
    tokenizer.pad_token = tokenizer.eos_token

    ## Dataset에 map 함수 적용
    def tokenize_and_label(batch):
        tokenized = tokenizer(
            batch["combined"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        ## 메구밍 발화의 시작 위치 탐색
        labels = tokenized["input_ids"].clone()
        
        for i, combined_text in enumerate(batch["combined"]):
            megumin_label = "megumin:"
            megumin_idx = combined_text.find(megumin_label)

            ## 메구밍 발화 이전 부분을 IGNORE_TOKEN_ID로 설정
            if megumin_idx != -1:
                tokenized_before_megumin = tokenizer(combined_text[:megumin_idx], max_length=1024, truncation=True)
                labels[i, :len(tokenized_before_megumin["input_ids"]) - 1] = -100
            else:
                labels[i, :] = -100  ## 메구밍 발화가 없다면 전체 무시

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels
        }

    ## Tokenize and label dataset
    return dataset.map(
        tokenize_and_label,
        batched=True,
        num_proc=4  ## 병렬 처리 (CPU 코어 수에 맞게 조정)
    )

## 데이터를 train/validation으로 나누기 (20%를 validation으로)
def split_train_validation(tokenized_dataset):
    train_test_split_result = tokenized_dataset.train_test_split(test_size=0.2, shuffle = True)
    return train_test_split_result['train'], train_test_split_result['test']

## Fine Tuning 설정 및 학습 진행
def fine_tune_model(tokenized_dataset):
    model.enable_input_require_grads()
    train_dataset, val_dataset = split_train_validation(tokenized_dataset)

    ## 학습 설정 -> 2e-5로 할거면 다음에는 9 epoch로 해볼 것
    training_args = TrainingArguments(
        output_dir="./megumin_results",     ## 모델이 저장될 경로
        learning_rate=2e-5,                 ## 초기 학습률 설정
        lr_scheduler_type="cosine",         ## 스케줄러 적용
        warmup_steps=900,                   ## Warmup 스텝
        overwrite_output_dir=True,          ## 경로 덮어쓰기
        num_train_epochs=10,                ## 학습할 epoch 수
        per_device_train_batch_size=8,      ## 배치 크기
        gradient_accumulation_steps=8,      ## 8 배치를 누적 후 그래디언트 업데이트
        save_steps=300,                     ## 저장 주기
        save_total_limit=1,                 ## 저장할 모델 체크포인트 수
        logging_dir="./megumin_logs",       ## 로그 저장 디렉토리
        logging_steps=100,                  ## 로그 기록 주기
        logging_first_step=False,           ## 첫 스텝 로깅 비활성화
        optim="paged_adamw_32bit",          ## QLoRA에 맞춘 32비트 AdamW 옵티마이저
        max_grad_norm=10,                   ## 그래디언트 노름을 10으로 제한
        evaluation_strategy="steps",        ## 검증 설정 (훈련 중 일정 주기로 검증)
        eval_steps=300,                     ## 검증 주기 (매 300 스텝마다)
        load_best_model_at_end=True,        ## 가장 좋은 모델 저장
)


    ## Trainer 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,           ## 검증 데이터 추가
        data_collator=DataCollatorWithProfile(tokenizer=tokenizer, profile_text=megumin_profile_metadata, prob_profile=0.3, mlm=False, pad_to_multiple_of=8),
        callbacks=[early_stopping_callback] ## Early Stopping 콜백 추가
    )

    ## 모델 Fine Tuning 진행
    trainer.train()

    ## 모델 저장
    model.save_pretrained("./model_qwen_7b_1122")
    tokenizer.save_pretrained("./tokenizer_qwen_7b_1122")