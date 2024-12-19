import pandas as pd
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from collections import Counter
import math
from itertools import islice

## 데이터 로드
answer = pd.read_csv('LLM_Evaluation.csv')[['category', 'question', 'answer']]
df = pd.read_excel('responses_qwen_rag.xlsx')[['category', 'question', 'ko_response']]

## 텍스트 정규화 함수 추가
def normalize_text(text):
    ## 공백 제거 및 소문자로 변환
    return ' '.join(text.strip().lower().split())

## ROUGE 스코어 계산기
rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

## perplexity 계산 함수
def calculate_perplexity(text):
    tokenized_text = text.split()
    if not tokenized_text:
        return 0.0  ## 빈 텍스트는 perplexity 0으로 처리
    word_freq = Counter(tokenized_text)
    total_words = len(tokenized_text)
    entropy = -sum((freq / total_words) * math.log2(freq / total_words) for freq in word_freq.values())
    perplexity = math.pow(2, entropy)
    return perplexity

## distinct 계산 함수
def calculate_distinct(text, n=1):
    tokenized_text = text.split()
    if not tokenized_text:
        return 0.0  ## 빈 텍스트는 distinct 점수 0으로 처리
    ngrams = list(zip(*(islice(tokenized_text, i, None) for i in range(n))))
    unique_ngrams = set(ngrams)
    return len(unique_ngrams) / len(ngrams) if ngrams else 0

## BLEURT 스코어 계산 함수
def calculate_bleurt(reference, response):
    inputs = tokenizer(reference, response, return_tensors="pt", truncation=True, padding=True)
    scores = model(**inputs).logits
    return scores.item()

## 결과 저장 리스트
evaluation_results = []

## 데이터프레임 비교 및 각 메트릭 계산
for _, row in df.iterrows():
    ## 참조(answer) 추출
    category = row['category']
    question = row['question']
    response = row['ko_response']
    reference = answer.drop_duplicates(subset=['category', 'question'])
    reference = answer[(answer['category'] == category) & (answer['question'] == question)]['answer'].values
    
    if len(reference) > 0:
        reference_text = normalize_text(reference[0])
        response = normalize_text(response)
        
        ## ROUGE 계산
        rouge_scores = rouge_scorer_instance.score(reference_text, response)
        
        ## BLEU 계산 수정
        if reference_text and response:
            bleu_score = sentence_bleu([reference_text.split()], response.split())
        else:
            bleu_score = 0.0

        ## BLEURT 계산
        bleurt_score = calculate_bleurt(reference_text, response)
        
        ## perplexity 계산
        perplexity_score = calculate_perplexity(response)
        
        ## distinct 계산
        distinct_1 = calculate_distinct(response, n=1)
        distinct_2 = calculate_distinct(response, n=2)
        
        ## 결과 저장
        evaluation_results.append({
            'category': category,
            'question': question,
            'rouge-1': rouge_scores['rouge1'].fmeasure,
            'rouge-2': rouge_scores['rouge2'].fmeasure,
            'rouge-L': rouge_scores['rougeL'].fmeasure,
            'bleu': bleu_score,
            'bleurt': bleurt_score,
            'perplexity': perplexity_score,
            'distinct-1': distinct_1,
            'distinct-2': distinct_2
        })
    else:
        ## 정답이 없는 경우 None 처리
        evaluation_results.append({
            'category': category,
            'question': question,
            'rouge-1': None,
            'rouge-2': None,
            'rouge-L': None,
            'bleu': None,
            'bleurt': None,
            'perplexity': None,
            'distinct-1': None,
            'distinct-2': None
        })

## 결과를 데이터프레임으로 변환
evaluation_df = pd.DataFrame(evaluation_results)

## 각 열의 합계 계산
column_sums = evaluation_df.iloc[:, 2:].sum()

## 소수점 둘째자리까지 포매팅
formatted_sums = column_sums.apply(lambda x: f"{x:.2f}")
print(formatted_sums)