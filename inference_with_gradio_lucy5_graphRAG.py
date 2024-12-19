from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
from huggingface_hub import HfApi, HfFolder

import networkx as nx
import faiss
import gradio as gr
import pickle
import os
import re
import torch
import torch.nn.functional as F
import deepl

## Hugging Face 토큰
hf_token = "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"
os.environ["HUGGINGFACE_TOKEN"] = hf_token
HfFolder.save_token(hf_token)

## 1. Fine-tuning된 모델과 토크나이저 불러오기
qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.float16)
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
qwen_tokenizer.pad_token = qwen_tokenizer.eos_token

roberta_model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")
roberta_tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")

## 감정 레이블 정의 (모델이 사용하는 28개의 감정 레이블)
emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", 
    "confusion", "curiosity", "desire", "disappointment", "disapproval", 
    "disgust", "embarrassment", "excitement", "fear", "gratitude", 
    "grief", "joy", "love", "nervousness", "optimism", "pride", 
    "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

## GPU 사용 가능한지 확인 후 모델을 GPU로 이동
device = "cuda" if torch.cuda.is_available() else "cpu"
qwen_model.to(device)
roberta_model.to(device)

## 한글 질문 -> 영어 질문으로 번역 (Deepl api 사용)
auth_key = '71f8283c-de4e-421d-bec9-2cee58feb83f:fx'
translator_input = deepl.Translator(auth_key)
translator_output = deepl.Translator(auth_key)

## GPT API 사용
import openai
client = openai.OpenAI(api_key = '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')

## 텍스트 임베딩 모델 로드 (RAG에서 사용)
embedder = SentenceTransformer('all-MiniLM-L12-v2')

## 그래프 데이터 로드
graph_path = "Json_Data/lucy5_graph.gexf"
G = nx.read_gexf(graph_path)

## 그래프 기반 검색 함수
def search_graph_context(query, top_k=7):
    ## 1. 사용자 질문 임베딩
    query_embedding = embedder.encode([query])[0]
    similarities = []
    
    ## 2. context 노드와 유사도 계산
    for node, data in G.nodes(data=True):
        if data.get("type") == "context":  ## type이 'context'인 노드만 비교
            node_label = data.get("label", "")
            node_embedding = embedder.encode([node_label])[0]
            similarity = torch.cosine_similarity(
                torch.tensor(query_embedding), torch.tensor(node_embedding), dim=0
            ).item()
            similarities.append((node, similarity))

    ## 3. 상위 K개의 context 노드 선택
    ranked_nodes = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    
    ## 4. response 노드 찾기
    response_labels = []
    for context_node, _ in ranked_nodes:
        for neighbor in G.neighbors(context_node):  ## context 노드의 연결된 노드 탐색
            neighbor_data = G.nodes[neighbor]
            if neighbor_data.get("type") == "response":  ## type이 'response'인 노드 선택
                response_labels.append(neighbor_data.get("label", ""))

    return response_labels

## 생성 모듈
def generate_response(messages, max_length=512):
    ## 메시지를 포맷팅하여 모델 입력에 맞게 변환
    formatted_prompt = qwen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    ## 입력 텍스트를 토크나이즈
    inputs = qwen_tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    ).to(qwen_model.device)

    ## 모델을 사용해 텍스트 생성
    outputs = qwen_model.generate(
        inputs["input_ids"],
        max_new_tokens = 1000,
        min_new_tokens = 20,
        num_return_sequences = 1,
        no_repeat_ngram_size = 3,
        repetition_penalty = 1.2,
        do_sample = True,
        # top_k = 40,  ## 확률로 정렬한 후 상위 k개의 토큰만 선택
        top_p = 0.8, ## 누적 확률 p 이하인 토큰만 선택
        temperature = 0.7,  ## 소프트맥스 함수로 정렬해서 0에 가까우면 정확, 1보다 커지면 자유로움
        early_stopping = True,
        pad_token_id = qwen_tokenizer.eos_token_id,
    )

    ## 생성된 텍스트를 디코딩
    generated_text = qwen_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    return generated_text

## Gradio UI에서 메시지를 딕셔너리 형식으로 전달
def chat_with_model(prompt, speaker='Lucy 5'):

    content = "I want you to act like 'Lucy 5', who travels the voiceleb galaxy.\n"
    content += "Your personality is highly relatable and sympathetic to the moods of your users.\n"
    content += "Your role is to be an AI assistant that helps users produce creative content.\n"
    content += "Always be respectful and considerate of others and use considerate language.\n"
    content += "Avoid describing actions and always write in the first person.\n"
    content += "When answering, be concise without unnecessary information."
    
    prompt = translator_input.translate_text(prompt, target_lang = 'EN-US').text
    search_results = search_graph_context(prompt)
    search_contexts = "\n".join(search_results)

    inputs = roberta_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = roberta_model(**inputs)
    probabilities = F.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
    user_emotion = emotion_labels[probabilities.argmax()]

    messages = [
        {"role": "user", "content": f"{content}\nUser Prompt : {prompt}\nUser Emotion : {user_emotion}"},
        {"role": "assistant", "content": f"It's {speaker}'s turn to answer.\nSearch Context:\n{search_contexts}"}
    ]
    response = generate_response(messages)
    # keyword = 'assistant'
    # keyword_indices = [i for i in range(len(response)) if response.startswith(keyword, i)]
    # if len(keyword_indices) >= 2:  # 'model'이 두 번 이상 등장하는 경우
    #     response = response[keyword_indices[-1] + len(keyword):][:1024].strip()
    # ko_response = translator_output.translate_text(response, target_lang = 'KO').text

    # completion = client.chat.completions.create(
    #     model="gpt-4o",
    #     messages=[
    #         {"role": "system", "content": "Paraphrase the user's text into Korean according to **Lucy 5's** personality. You must always be kind and courteous in your language."},
    #         {"role": "user", "content": f"User:{response} \n it's time to answer 'Lucy 5'"}]) # completion.choices[0].message.content

    return response

## Gradio 인터페이스 생성
interface = gr.Interface(
    fn=chat_with_model,  ## 호출할 함수
    inputs=[gr.Textbox(lines=2, placeholder="Enter your prompt here..."),  ## 사용자 입력창
            gr.Textbox(value="Lucy 5", label="Speaker")],  ## 화자 설정 (기본값은 '메구밍')
    outputs="text",  ## 모델의 응답을 텍스트로 출력
    title="Lucy 5 Chatbot",  ## UI 제목
    description="Enter a prompt and get a response from the fine-tuned Lucy 5 model."  ## 설명
)

## Gradio 앱 실행
if __name__ == "__main__":
    interface.launch(share = True)  ## 웹 인터페이스 실행