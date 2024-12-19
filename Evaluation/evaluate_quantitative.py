## evaluate_statistics.py
## 데이터의 통계적 정량 평가 코드

import pandas as pd
import math
from collections import Counter
from itertools import islice
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

## 텍스트 정규화 함수
def normalize_text(text):
    return ' '.join(text.strip().lower().split())

## perplexity 계산 함수
def calculate_perplexity(text):
    tokenized_text = text.split()
    if not tokenized_text:
        return 0.0
    word_freq = Counter(tokenized_text)
    total_words = len(tokenized_text)
    entropy = -sum((freq / total_words) * math.log2(freq / total_words) for freq in word_freq.values())
    return math.pow(2, entropy)

## distinct 계산 함수
def calculate_distinct(text, n=1):
    tokenized_text = text.split()
    if not tokenized_text:
        return 0.0
    ngrams = list(zip(*(islice(tokenized_text, i, None) for i in range(n))))
    unique_ngrams = set(ngrams)
    return len(unique_ngrams) / len(ngrams) if ngrams else 0

## BLEURT 스코어 계산 함수
def calculate_bleurt(reference, response, tokenizer, model):
    inputs = tokenizer(reference, response, return_tensors="pt", truncation=True, padding=True)
    scores = model(**inputs).logits
    return scores.item()

## 데이터 정량 평가 함수
def evaluate_responses(df, answer_df, tokenizer, model, rouge_scorer_instance):
    evaluation_results = []
    for _, row in df.iterrows():
        category, question, response = row['category'], row['question'], row['ko_response']
        reference = answer_df[(answer_df['category'] == category) & (answer_df['question'] == question)]['answer'].values

        if len(reference) > 0:
            reference_text = normalize_text(reference[0])
            response = normalize_text(response)

            rouge_scores = rouge_scorer_instance.score(reference_text, response)
            bleu_score = sentence_bleu([reference_text.split()], response.split()) if reference_text and response else 0.0
            bleurt_score = calculate_bleurt(reference_text, response, tokenizer, model)
            perplexity_score = calculate_perplexity(response)
            distinct_1 = calculate_distinct(response, n=1)
            distinct_2 = calculate_distinct(response, n=2)

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

    return pd.DataFrame(evaluation_results)

if __name__ == "__main__":
    ## 모델 및 토크나이저 초기화
    MODEL_NAME = "Elron/bleurt-base-512"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    ## 데이터 로드
    answer_df = pd.read_csv('LLM_Evaluation.csv')[['category', 'question', 'answer']]
    df = pd.read_excel('responses_qwen_rag.xlsx')[['category', 'question', 'ko_response']]

    ## 평가 수행
    evaluation_df = evaluate_responses(df, answer_df, tokenizer, model, rouge_scorer_instance)

    ## 결과 저장
    evaluation_df.to_excel('evaluation_results.xlsx', index=False)
    print("통계적 평가가 완료되었습니다. 결과가 'evaluation_results.xlsx'에 저장되었습니다.")

