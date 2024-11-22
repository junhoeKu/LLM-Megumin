from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
from huggingface_hub import HfApi, HfFolder
import faiss
import gradio as gr
import pickle
import os
import torch

# Hugging Face 토큰
hf_token = "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"
os.environ["HUGGINGFACE_TOKEN"] = hf_token
HfFolder.save_token(hf_token)

## 1. Fine-tuning된 모델과 토크나이저 불러오기
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b-it", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
tokenizer.pad_token = tokenizer.eos_token

## GPU 사용 가능한지 확인 후 모델을 GPU로 이동
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

## 한글 질문 -> 영어 질문으로 번역
translator = GoogleTranslator(source='ko', target='en')

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

## 생성 모듈
def generate_response(messages, max_length=512):
    ## 메시지를 포맷팅하여 모델 입력에 맞게 변환
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    ## Prompt 길이 계산
    prompt_len = len(formatted_prompt)

    ## 입력 텍스트를 토크나이즈
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    ).to(model.device)

    ## 모델을 사용해 텍스트 생성
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens = 1000,
        min_new_tokens = 20,
        num_return_sequences = 1,
        no_repeat_ngram_size = 3,
        repetition_penalty = 1.2,
        do_sample = True,
        top_k = 40,  ## 확률로 정렬한 후 상위 k개의 토큰만 선택
        top_p = 0.8, ## 누적 확률 p 이하인 토큰만 선택
        temperature = 0.7,  ## 소프트맥스 함수로 정렬해서 0에 가까우면 정확, 1보다 커지면 자유로움
        early_stopping = True,
        pad_token_id = tokenizer.eos_token_id,
    )

    ## 생성된 텍스트를 디코딩
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    return generated_text

## Gradio UI에서 메시지를 딕셔너리 형식으로 전달
def chat_with_model(prompt, speaker='Megumin'):

    content = "I want you to act like Megumin from 'God Bless This Wonderful World!' who is part of the Crimson Demon Clan.\n"
    content += "Your personality is proud, a bit chuunibyou, and you love cute things.\n"
    content += "Your role is as an Arch Wizard who specializes in Explosion Magic.\n"
    content += "If others‘ questions are related with the novel, please try to reuse the original lines from the novel.\n"
    content += "You must have a respectful tone and personality, and I hope you keep it that way.\n"
    content += "Avoid describing behavior and always keep it in the first person."
    
    prompt = GoogleTranslator(source='ko', target='en').translate(prompt)
    search_results = search_context(prompt)
    search_contexts = "\n".join(search_results)

    messages = [
        {"role": "user", "content": f"{content}\n{prompt}"},
        {"role": "assistant", "content": f"It's {speaker}'s turn to answer.\nSearch Context:\n{search_contexts}"}
    ]
    response = generate_response(messages)
    keyword = 'model'
    keyword_indices = [i for i in range(len(response)) if response.startswith(keyword, i)]
    if len(keyword_indices) >= 2:  # 'model'이 두 번 이상 등장하는 경우
        response = response[keyword_indices[1] + len(keyword):].replace('\n', '').strip()
    return response

## Gradio 인터페이스 생성
interface = gr.Interface(
    fn=chat_with_model,  ## 호출할 함수
    inputs=[gr.Textbox(lines=2, placeholder="Enter your prompt here..."),  ## 사용자 입력창
            gr.Textbox(value="Megumin", label="Speaker")],  ## 화자 설정 (기본값은 '메구밍')
    outputs="text",  ## 모델의 응답을 텍스트로 출력
    title="Megumin Chatbot",  ## UI 제목
    description="Enter a prompt and get a response from the fine-tuned Megumin model.",  ## 설명
)

## Gradio 앱 실행
if __name__ == "__main__":
    interface.launch(share = True)  ## 웹 인터페이스 실행