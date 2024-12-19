from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline\
    , AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from huggingface_hub import HfApi, HfFolder
import speech_recognition as sr
import soundfile as sf
from gtts import gTTS
import subprocess
import pyaudio
import faiss
import gradio as gr
import pickle
import os
import re
import torch
import torch.nn.functional as F
import pandas as pd
import deepl
from datetime import datetime

## Hugging Face 토큰
hf_token = ""
os.environ["HUGGINGFACE_TOKEN"] = hf_token
HfFolder.save_token(hf_token)

## 기존 데이터프레임 불러오기
df = pd.read_csv('Json_Data/lucy5_answer.csv')

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
auth_key = ''
translator_input = deepl.Translator(auth_key)
translator_output = deepl.Translator(auth_key)

## GPT API 사용
import openai
client = openai.OpenAI(api_key = '')

## 음성을 텍스트로 변환 (STT)
def speech_to_text(audio_path=None):
    r = sr.Recognizer()
    try:
        if audio_path:  ## 오디오 파일이 입력된 경우
            with sr.AudioFile(audio_path) as source:
                audio = r.record(source)
        else:  ## 마이크를 통해 실시간 음성 입력
            with sr.Microphone() as source:
                print("말하세요...")
                audio = r.listen(source, timeout=30, phrase_time_limit=30)
        
        ## Google Web Speech API를 사용해 음성 인식
        text = r.recognize_google(audio, language='ko-KR')
        return text
    except sr.UnknownValueError:
        return "음성을 인식할 수 없습니다."
    except sr.RequestError as e:
        return f"Google Speech Recognition 서비스에 문제가 발생했습니다: {e}"
    except Exception as e:
        return f"STT 오류 발생: {e}"

## 텍스트를 음성으로 변환 (TTS)
def text_to_speech(text, lang='ko'):
    tts = gTTS(text=text, lang=lang)
    file_path = "output.mp3"
    tts.save(file_path)
    return file_path

## 텍스트 임베딩 모델 로드 (RAG에서 사용)
embedder = SentenceTransformer('all-MiniLM-L12-v2')

## FAISS 인덱스와 응답 데이터 로드
index_path = "Json_Data/lucy5_context_index.faiss"
response_path = "Json_Data/lucy5_responses.pkl"
index = faiss.read_index(index_path)
with open(response_path, "rb") as f:
    responses = pickle.load(f)

## 검색 모듈 (RAG)
def search_context(query, top_k=10):
    ## 1. 검색 전 프로세스 : 전처리 + 메타데이터 추가 + 임베딩 생성
    query = re.sub(r'\s+', ' ', query.strip())  ## 쿼리 전처리
    query = re.sub(r'[^\w\s]', '', query)  ## 쿼리 전처리
    query_embedding = embedder.encode([query])  ## 쿼리 임베딩 생성

    ## 2. 유사한 컨텍스트 검색
    distances, indices = index.search(query_embedding, top_k)
    results = [responses[i] for i in indices[0]]

    # 3. 검색 후 프로세스 : 재순위화
    ranked_results = sorted(zip(results, distances[0]), key=lambda x: x[1], reverse=True)
    refined_results = [result[0] for result in ranked_results]  ## 정렬된 결과만 추출

    return refined_results

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
def chat_with_text_or_audio(input_text, audio):
    global df

    if audio is not None:
        try:
            prompt = speech_to_text(audio)
        except Exception as e:
            return f"STT 오류 발생: {e}", None

    elif input_text.strip():
        prompt = input_text

    else:
        return "입력값이 없습니다.", None

    content = "I want you to act like 'Lucy 5', who travels the voiceleb galaxy.\n"
    content += "Your personality is highly relatable and sympathetic to the moods of your users.\n"
    content += "Your role is to be an AI assistant that helps users produce creative content.\n"
    content += "Always be respectful and considerate of others and use considerate language.\n"
    content += "Avoid describing actions and always write in the first person.\n"
    content += "When answering, be concise without unnecessary information."

    ko_prompt = prompt
    prompt = translator_input.translate_text(prompt, target_lang = 'EN-US').text
    search_results = search_context(prompt)
    search_contexts = "\n".join(search_results)

    inputs = roberta_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = roberta_model(**inputs)
    probabilities = F.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
    user_emotion = emotion_labels[probabilities.argmax()]

    messages = [
        {"role": "user", "content": f"{content}\nUser Prompt : {prompt}\nUser Emotion : {user_emotion}"},
        {"role": "assistant", "content": f"It's Lucy 5's turn to answer.\nSearch Context:\n{search_contexts}"}
    ]
    response = generate_response(messages)
    keyword = 'assistant'
    keyword_indices = [i for i in range(len(response)) if response.startswith(keyword, i)]
    if len(keyword_indices) >= 2:  # 'model'이 두 번 이상 등장하는 경우
        response = response[keyword_indices[-1] + len(keyword):][:1024].strip()

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Paraphrase the user's text into Korean according to **Lucy 5's** personality. You must always be kind and courteous in your language."},
            {"role": "user", "content": f"User:{response} \n it's time to answer 'Lucy 5'"}]) # completion.choices[0].message.content
    
    ko_response = completion.choices[0].message.content

    ## TTS 생성
    audio_path = text_to_speech(ko_response)

    ## 데이터프레임 업데이트
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_row = {"time": current_time, "question": ko_prompt, "answer": ko_response}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv("Json_Data/lucy5_answer.csv", index=False)

    return ko_response, audio_path

## Gradio 인터페이스 생성
interface = gr.Interface(
    fn=chat_with_text_or_audio,  ## 호출할 함수
    inputs=[
        gr.Textbox(label="Enter your text", lines=2, placeholder="Type your question here..."),
        gr.Audio(sources = ['microphone', 'upload'], type="filepath", streaming = False, label="Or record your question", format="wav")
    ],
    outputs=[
        gr.Textbox(label="Response"),  ## 텍스트 출력
        gr.Audio(label="TTS Output")  ## 오디오 출력
    ],
    title="Lucy 5 Chatbot with TTS",
    description="Enter a prompt and get a response from Lucy 5 with text-to-speech functionality."
)

## Gradio 앱 실행
if __name__ == "__main__":
    interface.launch(share=True)
