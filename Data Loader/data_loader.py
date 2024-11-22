## data_loader.py

import json
from datasets import Dataset
from transformers import AutoTokenizer
import re
import os

## 전처리 함수 정의
def preprocess_text(text):
    text = re.sub(r'\.{3,}', '...', text.replace('…', '.'))  ## '…'을 '...'로 변환
    text = re.sub(r'[^>]*:\s*', '', text)
    text = text.replace('─', '').replace('―', '').replace('-', '')  ## 불필요한 대시 제거
    text = text.replace('『', ' ').replace('』', ' ')  ## 특수 괄호 제거
    text = text.replace('【', '\'').replace('】', '\'')  ## 대괄호를 큰따옴표로 변환
    text = text.replace('「', '\'').replace('」', '\'')  ## 작은따옴표를 큰따옴표로 변환
    text = text.replace('ㆍ', ' ').replace('(爆裂道)', '')  ## 'ㆍ'를 공백으로 변환 및 '(爆裂道)' 제거
    text = text.replace('ㄸ', '또')  ## 'ㄸ'을 '또'로 변환
    
    return text

## 장면(context)에서 메구밍 발화를 찾고, 그 앞의 3문장을 추출하는 함수
def extract_scene_context(scene, megumin_line):
    ## 따옴표(")로 장면을 구분
    parts = preprocess_text(scene).split("\"")
    
    ## 따옴표로 구분한 각 파트의 공백 제거
    parts = [part.strip() for part in parts if part.strip()]
    
    ## 메구밍 발화와 일치하는 부분을 찾음
    if megumin_line in scene:
        index = next((i for i, s in enumerate(parts) if megumin_line in s), None)
        if index is not None:
            ## 메구밍 발화 앞의 바로 직전 나레이션만 추출
            if index > 0 and index % 2 == 1:  ## 메구밍 발화의 바로 앞에 나레이션이 있어야 함
                return parts[index - 1]  ## 메구밍 발화 바로 앞에 위치한 나레이션을 반환
    
    return ""  ## 일치하는 문장이 없으면 빈 문자열 반환

## 데이터셋 불러오기 및 전처리 적용
def load_megumin_dialogues_to_blessing(directory_path):
    dialogues = []  ## 문맥과 메구밍 대사(응답)를 함께 저장할 리스트
    context = []    ## 문맥을 저장할 리스트

    ## 디렉토리 내 모든 파일을 탐색
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):  ## 확장자가 .json인 파일만 처리
            file_path = os.path.join(directory_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                ## 각 파일에서 대사와 화자 정보를 필터링
                for session, content in data.items():
                    dialogue = content["dialogue"]
                    scene = content.get("scene", "")  ## 장면 정보가 없을 수 있으므로 get 사용
                    current_context = []  ## 현재 문맥을 추적하는 리스트
                    megumin_responses = []  ## 메구밍의 여러 발화를 합칠 리스트
                    
                    for line in dialogue:
                        if " : " in line:  ## 화자와 대사가 구분되어 있는지 확인
                            speaker, text = line.split(" : ", 1)  ## 화자와 대사를 분리
                            speaker = speaker.replace("\"", "").strip()
                            text = text.replace("\"", "").strip()  ## 전처리 적용
                            
                            ## 메구밍이 아닌 화자의 발화를 문맥에 추가
                            if speaker != "메구밍":
                                ## 이전 메구밍 발화가 있으면 저장
                                if megumin_responses:
                                    ## context에 scene 정보 추가
                                    scene_context = extract_scene_context(scene, megumin_responses[-1])
                                    dialogues.append({
                                        "scene_context": preprocess_text(scene_context),  ## 장면에서 추출한 문맥
                                        "context": preprocess_text(" ".join(current_context[-3:])),  ## 최근 3개의 문맥만 저장
                                        "response": preprocess_text(" ".join(megumin_responses))  ## 메구밍의 대사들을 하나로 합침
                                    })
                                    megumin_responses = []  ## 메구밍 발화 초기화
                                    current_context = []  ## 문맥도 초기화
                                
                                current_context.append(f"{speaker} : {text}")
                            
                            ## 메구밍의 발화가 나온 경우
                            elif speaker == "메구밍":
                                megumin_responses.append(text)  ## 메구밍 발화 합치기
                                
                    ## 파일이 끝나면 남은 메구밍 발화 저장
                    if megumin_responses:
                        ## context에 scene 정보 추가
                        scene_context = extract_scene_context(scene, megumin_responses[-1])
                        dialogues.append({
                            "scene_context": preprocess_text(scene_context),  ## 장면에서 추출한 문맥
                            "context": preprocess_text(" ".join(current_context[-3:])),  ## 최근 3개의 문맥만 저장
                            "response": preprocess_text(" ".join(megumin_responses))  ## 마지막 메구밍 발화들
                        })
    
    return dialogues

## 데이터셋 불러오기
def load_megumin_dialogues_to_flame(file_path):
    dialogues = []  ## 문맥과 메구밍 대사(응답)를 함께 저장할 리스트
    current_context = []  ## 문맥을 저장할 리스트
    megumin_responses = []  ## 메구밍의 연속된 발화를 저장할 리스트
    
    ## JSON 파일을 불러옴
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
        ## "dialogue" 키 내의 대사와 화자 정보를 처리
        for dialogue in data["dialogue"]:
            if ":" in dialogue:  ## 화자와 대사가 구분되어 있는지 확인
                speaker, text = dialogue.split(':', 1)  ## 화자와 대사를 분리
                speaker = speaker.strip()
                text = preprocess_text(text.strip())  ## 전처리 적용
                
                ## 메구밍이 아닌 화자의 발화를 문맥에 추가
                if speaker != "메구밍":
                    ## 이전에 메구밍 발화가 있었다면 저장
                    if megumin_responses:
                        dialogues.append({
                            "scene_context" : "",
                            "context": " ".join(current_context[-3:]),  # 최근 3개의 문맥만 저장
                            "response": " ".join(megumin_responses)  # 메구밍의 발화를 하나로 합침
                        })
                        megumin_responses = []  # 메구밍 발화 초기화
                        current_context = []  # 문맥도 초기화
                    
                    current_context.append(f"{speaker} : {text}")
                
                ## 메구밍의 발화가 나온 경우, 발화를 모아서 저장
                elif speaker == "메구밍":
                    megumin_responses.append(text)
        
        ## 마지막에 남은 메구밍의 발화가 있으면 처리
        if megumin_responses:
            dialogues.append({
                "scene_context" : "",
                "context": " ".join(current_context[-3:]),  # 최근 3개의 문맥만 저장
                "response": " ".join(megumin_responses)
            })
    
    return dialogues

## 데이터셋 불러오기
def load_megumin_qa_dialogues(file_path):
    dialogues = []  ## 문맥과 메구밍 대사(응답)를 함께 저장할 리스트
    megumin_responses = []  ## 메구밍의 연속된 발화를 저장할 리스트
    
    ## JSON 파일을 불러옴
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
        ## Q, A 쌍을 처리
        for dialogue in data:
            ## Q는 문맥으로 사용
            context = dialogue["Q"].strip()
            
            ## A는 메구밍의 응답으로 사용
            response = dialogue["A"].strip()
            megumin_responses.append(response)  # 메구밍의 응답을 모음
            
            ## 메구밍의 여러 발화를 하나로 합쳐 저장
            dialogues.append({
                "scene_context" : "사용자와 메구밍이 대화하는 상황",
                "context": context,  ## Q를 문맥으로 사용
                "response": " ".join(megumin_responses)  ## 여러 메구밍 응답을 합침
            })
            megumin_responses = []  # 응답 초기화
    
    return dialogues

## 깃허브 데이터 전용 데이터로더 함수
def extract_github_dialogues(file_path):
    
    ## JSON 파일을 불러옴   
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    dialogues = []  # 최종 데이터를 저장할 리스트
    current_context = []  # 현재 문맥을 저장할 리스트
    megumin_responses = []  # 메구밍 발화를 저장할 리스트
    
    for idx, line in enumerate(data):
        line = line.strip()
        if ":" in line:  # 대사 구분자인 ':'이 있는지 확인
            try:
                speaker, dialogue = line.split(":", 1)
                speaker = speaker.strip()
                dialogue = dialogue.strip()
                
                if speaker == "메구밍":  # 메구밍의 발화인 경우
                    # 메구밍 발화가 연속될 경우 발화를 이어줌
                    megumin_responses.append(dialogue)
                    
                    # scene_context는 메구밍 발화 앞의 나레이션으로 설정 (발화는 포함되지 않음)
                    scene_context = ""
                    if len(current_context) > 0:
                        last_context = current_context[-1]
                        if ":" not in last_context:  # 마지막 문맥이 나레이션일 경우
                            scene_context = last_context
                    
                    # context는 마지막 3개의 문맥을 사용 (발화와 나레이션을 모두 포함)
                    context = " ".join(current_context[-3:])

                else:
                    # 메구밍 발화가 끝났다면 이전 발화들을 저장
                    if megumin_responses:
                        dialogues.append({
                            "scene_context": scene_context,
                            "context": context,
                            "response": " ".join(megumin_responses)  # 메구밍 발화들을 하나로 이어줌
                        })
                        megumin_responses = []  # 메구밍 발화 리스트 초기화
                        current_context = []  # 문맥도 초기화
                    
                    # 메구밍이 아닌 발화를 문맥에 추가
                    current_context.append(f"{speaker} : {dialogue}")
                    
            except ValueError:
                print(f"Error processing line {idx}: '{line}'")  # 에러 발생 시 로그 출력
        else:
            current_context.append(line)  # 나레이션일 경우 문맥에 추가

    # 마지막으로 남은 메구밍 발화를 처리
    if megumin_responses:
        dialogues.append({
            "scene_context": scene_context,
            "context": context,
            "response": " ".join(megumin_responses)
        })
    
    return dialogues

## 생성된 데이터 저장 함수
def save_dialogues_as_json(dialogues, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dialogues, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    blessing_directory_path = "Json_Data/blessing_folder"
    flame_directory_path = "Json_Data/flame_dialogue.json"
    after_directory_path = "Json_Data/after_dialogue.json"
    movie_directory_path = "Json_Data/movie_dialogue.json"
    github_directory_path = "Json_Data/github_dialogue.json"
    qa_directory_path = "Json_Data/dialogues.json"

    blessing_dialogues = load_megumin_dialogues_to_blessing(blessing_directory_path)
    flame_dialogues = load_megumin_dialogues_to_flame(flame_directory_path)
    after_dialogues = load_megumin_dialogues_to_flame(after_directory_path)
    movie_dialogues = load_megumin_dialogues_to_flame(movie_directory_path)
    github_dialogue = extract_github_dialogues(github_directory_path)
    qa_dialogues = load_megumin_qa_dialogues(qa_directory_path)

    blessing_dialogues.extend(flame_dialogues)
    blessing_dialogues.extend(after_dialogues)
    blessing_dialogues.extend(movie_dialogues)
    blessing_dialogues.extend(github_dialogue)
    blessing_dialogues.extend(qa_dialogues)

    save_dialogues_as_json(blessing_dialogues, 'output.json')
