## inference_dataframe.py

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer
from huggingface_hub import HfApi, HfFolder
from tqdm import tqdm
import pandas as pd
import faiss
import gradio as gr
import pickle
import os
import torch

hf_token = ""
os.environ["HUGGINGFACE_TOKEN"] = hf_token
HfFolder.save_token(hf_token)

## GPU 사용 가능한지 확인 후 모델을 GPU로 이동
device = "cuda" if torch.cuda.is_available() else "cpu"

## 1. Fine-tuning된 모델과 토크나이저 불러오기
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

## GPU 사용 가능한지 확인 후 모델을 GPU로 이동
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

## 텍스트 임베딩 모델 로드 (RAG에서 사용)
embedder = SentenceTransformer('all-MiniLM-L12-v2')

## FAISS 인덱스와 응답 데이터 로드
index_path = "Json_Data/megumin_context_index.faiss"
response_path = "Json_Data/megumin_responses.pkl"
index = faiss.read_index(index_path)
with open(response_path, "rb") as f:
    responses = pickle.load(f)

## 검색 모듈 (RAG)
def search_context(query, top_k=10):
    ## 쿼리 임베딩 생성
    query_embedding = embedder.encode([query])

    ## 유사한 컨텍스트 검색
    distances, indices = index.search(query_embedding, top_k)
    results = [responses[i] for i in indices[0]]

    return results

## 3. Inference 함수 정의
def generate_response(prompt, speaker='메구밍', max_length=1024):
    content = "I want you to act like Megumin from 'God Bless This Wonderful World!' who is part of the Crimson Demon Clan.\n"
    content += "Your personality is proud, a bit chuunibyou, and you love cute things.\n"
    content += "Your role is as an Arch Wizard who specializes in Explosion Magic.\n"
    content += "If others‘ questions are related with the novel, please try to reuse the original lines from the novel.\n"
    content += "You must have a respectful tone and personality, and I hope you keep it that way.\n"
    content += "Avoid describing behavior and always keep it in the first person."

    search_results = search_context(prompt)
    search_contexts = "\n".join(search_results)

    messages = [
        {"role": "user", "content": f"{content}\n{prompt}"},
        {"role": "assistant", "content": f"It's {speaker}'s turn to answer.\nSearch Context:\n{search_contexts}"}
    ]

    ## Chat 템플릿을 적용하여 입력 데이터 생성 (여기까지는 문자열)
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    formatted_prompt_len = len(formatted_prompt)

    ## 생성된 문자열을 토크나이즈하여 모델에 전달 가능한 입력 형식으로 변환
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_attention_mask=True
    ).to(device)

    ## 3. 모델을 사용해 텍스트 생성
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens = 1000,  ## 생성할 최대 문장 길이
        min_new_tokens = 20,  ## 생성할 최소 문장 길이
        num_return_sequences = 1,  ## 한 번에 두 개의 문장 생성
        no_repeat_ngram_size = 3,  ## 반복 토큰 방지
        repetition_penalty = 1.2,  ## 반복 문장 방지
        do_sample = True,  ## 샘플링 활성화 (더 다양하게 생성)
        top_k = 40,
        top_p = 0.8,
        temperature = 0.7,
        early_stopping = True,
        pad_token_id=tokenizer.eos_token_id,  ## 패딩 토큰을 EOS로 설정
    )

    ## 생성된 텍스트를 디코딩
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    keyword = 'assistant'
    keyword_indices = [i for i in range(len(generated_text)) if generated_text.startswith(keyword, i)]
    if len(keyword_indices) >= 2:  # 'model'이 두 번 이상 등장하는 경우
        generated_text = generated_text[keyword_indices[2] + len(keyword):].replace('\n', ' ').replace('  ', ' ').strip()
    return generated_text

## 4. DataFrame 처리 함수 정의
def generate_responses_for_dataframe(df, question_column):
    responses = []
    for question in tqdm(df[question_column], desc="Generating Responses", unit="question"):
        response = generate_response(question)  ## 모델로 답변 생성
        responses.append(response)
    
    # 새로운 열에 답변 저장
    df['response'] = responses
    return df

if __name__ == '__main__':
    df = pd.read_csv('LLM_Evaluation.csv')

    ## 5. 질문에 대해 답변 생성 및 새로운 열 추가
    df_with_responses = generate_responses_for_dataframe(df, 'question')
    df_with_responses.to_excel('responses.xlsx', index=False)
