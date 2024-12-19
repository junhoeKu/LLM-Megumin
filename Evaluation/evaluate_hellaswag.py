## evaluate_hellaswag.py
## hellaswag 평가 코드

import pandas as pd
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from huggingface_hub import HfApi, HfFolder
import torch
import faiss
import pickle
from tqdm import tqdm

## Hugging Face 토큰
hf_token = "토큰을 입력하세요"
os.environ["HUGGINGFACE_TOKEN"] = hf_token
HfFolder.save_token(hf_token)

## 모델 및 토크나이저 로드
MODEL_NAME = "google/gemma-2-9b-it"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

## 텍스트 임베딩 모델 및 FAISS 설정
embedder = SentenceTransformer('all-MiniLM-L12-v2')
index_path = "Json_Data/megumin_context_index.faiss"
response_path = "Json_Data/megumin_responses.pkl"

## FAISS 인덱스 로드
index = faiss.read_index(index_path)
with open(response_path, "rb") as f:
    responses = pickle.load(f)

## 검색 모듈 (RAG)
def search_context(query, top_k=7):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    results = [responses[i] for i in indices[0]]
    return results

## 답변 생성 모듈
def generate_response(messages):
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=150,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

## 가장 일치하는 엔딩 찾기
def find_best_ending(response, endings):
    scores = []
    for ending in endings:
        ## 단순하게 LLM 출력과 엔딩의 유사도 비교 (여기서는 단순한 텍스트 길이 기반으로도 가능)
        score = response.count(ending)  ## LLM 출력에서 엔딩이 등장한 횟수를 기반으로
        scores.append(score)
    return scores.index(max(scores))  ## 가장 일치하는 엔딩의 인덱스 반환

## 평가 함수
def evaluate_llm_with_multiclass(jsonl_path, model, tokenizer, device):
    results = []
    with open(jsonl_path, "r") as f:
        data = [json.loads(line) for line in f][:1000]

    for row in tqdm(data, total=len(data), desc="Evaluating"):
        context = row.get("ctx", "")
        endings = row.get("endings", [])
        label = row.get("label", "")

        ## RAG 검색
        context_results = search_context(context)
        rag_context = "\n".join(context_results)

        ## LLM 프롬프트 생성
        endings_text = "\n".join([f"{i}: {ending}" for i, ending in enumerate(endings)])
        messages = [
            {"role": "user", "content": f"Context:\n{rag_context}\n\n{context}\nHere are four options:\n{endings_text}\n\nChoose the most appropriate option by its index (0, 1, 2, or 3)."},
            {"role": "assistant", "content": "The best option is:"},
        ]
        response = generate_response(messages)

        ## LLM 답변에서 가장 적절한 엔딩 인덱스 추출
        prediction = find_best_ending(response, endings)
        results.append((context, label, prediction))
    
    ## 결과 저장
    results_df = pd.DataFrame(results, columns=["Context", "Label", "Prediction"])
    return results_df

if __name__ == '__main__':
    ## 데이터 로드 및 평가
    file_path = "Json_Data/hellaswag_train.jsonl"
    evaluation_results = evaluate_llm_with_multiclass(file_path, model, tokenizer, device)

    ## 평가 결과 저장
    output_file = "hellaswag_multiclass_evaluation_results.csv"
    evaluation_results.to_csv(output_file, index=False)
    print(f"Evaluation results saved to {output_file}")
