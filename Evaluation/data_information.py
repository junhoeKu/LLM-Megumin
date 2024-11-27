import json
from collections import Counter
from sentence_transformers import SentenceTransformer


if __name__ == '__main__':

    ## Json 파일 로드
    with open('../LLM/augmented_data.json', "r", encoding="utf-8") as f:
        data = json.load(f)

    ## 토크나이저 로드 (적절한 모델 선택)
    tokenizer = SentenceTransformer("all-MiniLM-L12-v2")
    ## 초기화
    dialogue_count = 0
    utterance_count = 0
    total_tokens = 0
    total_words = 0

    ## 데이터 정량화 계산
    for dialogue in data:
        dialogue_count += 1  ## 각 JSON 객체는 하나의 대화로 간주
        for key in ["scene_context", "context", "response"]:
            text = dialogue.get(key, "")  ## 키가 없을 경우 기본값으로 빈 문자열
            if not isinstance(text, str) or not text.strip():
                continue  ## 텍스트가 유효하지 않으면 건너뜀

            utterance_count += 1  ## 발화 수 증가
            total_tokens += len(tokenizer.tokenize(text))  ## 토큰 수 계산
            total_words += len(text.split())  ## 단어 수 계산

    ## 결과 출력
    print(f"Dialogue count: {dialogue_count}")
    print(f"Utterance count: {utterance_count}")
    print(f"Total tokens: {total_tokens}")
    print(f"Total words: {total_words}")