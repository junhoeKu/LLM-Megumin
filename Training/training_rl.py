import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import pandas as pd
from datasets import Dataset
import os

## GPU 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

## PPO 설정
ppo_config = PPOConfig(
    model_name="./model_gemma2_9b_1021_2",  # 미리 학습된 모델 사용
    learning_rate=1.41e-5,
    batch_size=128
)

# 토크나이저와 모델 로드
tokenizer = AutoTokenizer.from_pretrained("./tokenizer_gemma2_9b_1021")
tokenizer.pad_token = tokenizer.eos_token

# Value Head가 포함된 모델 로드
model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_config.model_name)
model.to("cuda" if torch.cuda.is_available() else "cpu")

# PPO 훈련자 설정
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    tokenizer=tokenizer
)

## 사용자 피드백을 기반으로 강화학습
def get_user_feedback(response):
    """ 
    사용자 피드백을 실시간으로 제공하는 함수.
    이 함수에서 사용자는 응답을 보고 0~100 사이의 점수를 제공할 수 있음.
    """
    print(f"모델의 답변: {response}")
    
    while True:
        try:
            feedback = int(input("답변에 대한 점수를 0에서 100 사이의 정수로 입력하세요: "))
            if 0 <= feedback <= 100:
                return feedback
            else:
                print("0에서 100 사이의 정수를 입력하세요.")
        except ValueError:
            print("유효한 숫자를 입력하세요.")

## Inference 함수 정의
def generate_response(prompt, max_length=400):
    formatted_prompt = (
        "너는 소설 '이 멋진 세계에 축복을!'의 등장인물 메구밍이야.\n"
        "메구밍은 홍마족의 자존심 강한 아크위자드이며, 유일하게 폭렬마법만 사용할 수 있는 마법사다.\n"
        "그녀의 성격은 중2병적이며 자존심이 강하고, 귀여운 것을 좋아하지만, 동료를 진심으로 아낀다.\n"
        "[일반] \"" + f"{prompt}" + "\" [메구밍]"
        "\n메구밍, [일반] 토큰과 [메구밍] 토큰 사이의 텍스트에 답해."
        "\n귀엽고 자신감 넘치며, 중2병적인 대답을 '존댓말'로 해줘."
    )
    prompt_len = len(formatted_prompt)

    ## 입력 텍스트를 토크나이즈
    inputs = tokenizer(formatted_prompt,
                       return_tensors="pt",
                       padding=True,
                       return_attention_mask=True).to(device)

    ## 모델을 사용해 텍스트 생성
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,  ## 생성할 최대 토큰 길이
        num_return_sequences=1,  ## 한 번에 하나의 문장 생성
        no_repeat_ngram_size=3,  ## 반복 방지
        do_sample=True,  ## 샘플링 활성화 (더 다양하게 생성)
        top_k=40,  ## top-k 샘플링
        top_p=0.95,  ## top-p (nucleus) 샘플링
        temperature=0.7,  ## 생성의 랜덤성 조정
        pad_token_id=tokenizer.eos_token_id,  ## 패딩 토큰을 EOS로 설정
    )

    ## 생성된 텍스트를 디코딩
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)[prompt_len:].strip()

    return generated_text

## 강화학습을 위한 데이터프레임 처리
def generate_responses_for_dataframe(df, question_column):
    responses = []
    for question in df[question_column]:
        response = generate_response(question)  # 모델로 답변 생성
        responses.append(response)

    # 새로운 열에 답변 저장
    df['response'] = responses
    return df

## PPO 학습 및 사용자 피드백 적용 함수
def train_with_feedback(df, question_column):
    for index, row in df.iterrows():
        question = row[question_column]
        
        # 모델이 답변 생성
        response = generate_response(question)
        
        # 사용자 피드백 받기
        reward = get_user_feedback(response)
        
        # 보상 기반으로 PPO 모델 업데이트
        inputs = tokenizer(question, return_tensors="pt", padding=True).to(device)
        query_tensors = inputs["input_ids"]
        response_tensors = tokenizer(response, return_tensors="pt").to(device)["input_ids"]
        rewards = torch.tensor([reward], dtype=torch.float).to(device)
        
        # PPO 업데이트 - 리스트로 변환하여 전달
        ppo_trainer.step([query_tensors], [response_tensors], [rewards])

    # 업데이트된 응답을 DataFrame에 저장
    df = generate_responses_for_dataframe(df, question_column)
    return df

if __name__ == "__main__":
    # CSV 파일로부터 데이터를 읽어옴
    data = {
        'question': [
            "메구밍, 너의 가장 강력한 마법은 뭐야?",
            "메구밍, 카즈마에 대해서 어떻게 생각해?",
            "너는 왜 폭렬마법만 고집하는 거야?",
            "너에 대해 소개해줘.",
            "한국의 수도가 어디야?"
        ]
    }

    df = pd.DataFrame(data)

    # 학습과 사용자 피드백 적용
    df_with_responses = train_with_feedback(df, 'question')

    # 강화학습 결과를 CSV 파일로 저장
    output_file = "responses_with_feedback.csv"
    df_with_responses.to_csv(output_file, index=False)
    print(f"결과를 {output_file}에 저장했습니다.")

    # 강화학습된 모델 저장
    model_save_path = "./ppo_finetuned_model_1021_2"
    tokenizer_save_path = "./ppo_finetuned_tokenizer_1021_2"

    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(tokenizer_save_path)

    print(f"강화학습된 모델이 {model_save_path}에 저장되었습니다.")
    print(f"강화학습된 토크나이저가 {tokenizer_save_path}에 저장되었습니다.")
