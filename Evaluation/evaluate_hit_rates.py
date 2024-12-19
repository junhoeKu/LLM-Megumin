## evaluate_hit_rates.py
## LLM 답변의 Hit Rates를 평가하는 코드

import pandas as pd

## 키워드 매칭 및 성공 여부 계산 함수
def calculate_keyword_hits(df, keywords_column, response_column):
    df = df.copy()

    ## 키워드 매칭 개수 계산
    df["keyword_hit_count"] = df.apply(
        lambda row: sum(1 for kw in row[keywords_column] if kw in row[response_column]), axis=1)

    ## 성공 여부 판단 (50% 이상의 키워드 포함 시 성공)
    df["success"] = df.apply(
        lambda row: 1 if row["keyword_hit_count"] >= len(row[keywords_column]) / 2 else 0,axis=1)

    return df

## HIT RATE 계산 함수
def calculate_hit_rate(df):
    return df["success"].mean()

## 메인 함수
def main(input_file, output_file, keywords):
    ## 데이터 로드
    df = pd.read_excel(input_file)

    ## 키워드 추가
    df["keyword"] = keywords

    ## 키워드 매칭 및 성공 여부 계산
    df = calculate_keyword_hits(df, keywords_column="keyword", response_column="ko_response")

    ## HIT RATE 계산
    hit_rate = calculate_hit_rate(df)
    print(f"HIT RATE: {hit_rate:.2%}")

    ## 결과 저장
    df.to_excel(output_file, index=False)

if __name__ == "__main__":
    ## 입력 파일과 출력 파일 경로 설정
    input_file = "responses_1126.xlsx"
    output_file = "evaluated_responses.xlsx"

    ## 키워드 리스트 정의
    keyword = [
        ['언니', '봉인', '폭렬 마법', '월버그', '마왕군'],
        ['고향', '홍마족', '강한', '마왕군', '아크위저드'],
        ['액셀', '초보', '카즈마', '아쿠아', '마을'],
        ['물', '알칸레티아', '마왕군', '한스', '슬라임'],
        ['모험가 카드', '상급', '스킬 포인트', '카즈마', '폭렬 마법'],
        ['마왕', '결계', '폭렬 마법', '몇', '토벌'],
        ['카즈마', '목욕', '어린', '부끄', '융융'],
        ['춈스케', '암컷', '고양이', '폭렬 마법', '사역마'],
        ['홍마족', '바코드', '엉덩이', '부끄', '카즈마'],
        ['디스트로이어', '아쿠아', '폭렬 마법', '마법 결계', '마력'],

        ['메구밍', '홍마족', '아크위저드', '폭렬 마법', '최강'],
        ['카즈마', '소중한', '좋아', '받아', '동료'],
        ['아쿠아', '여신', '민폐', '엉뚱', '동료'],
        ['다크니스', '크루세이더', '방어력', '마조', '동료'],
        ['가족', '효이자부로', '유이유이', '코멧코', '춈스케'],
        ['작명 센스', '춈스케', '츈츈마루', '홍마족', '이름'],
        ['머리', '홍마족', '폭렬 마법', '누가', '이상'],
        ['융융', '라이벌', '친구', '든든', '자칭'],
        ['귀여운', '춈스케', '젤 킹', '파오리', '좋아'],
        ['익스플로전', '폭렬 마법', '마법', '혼돈', '암흑'],

        ['카즈마', '솔직', '의지', '좋아', '따뜻'],
        ['폭렬 마법', '다른', '상급', '강력', '마법'],
        ['폭렬 마법', '강력한', '잔뜩', '상대', '몬스터'],
        ['뿌듯', '폭렬 마법', '100점', '카즈마', '120점'],
        ['홍마족', '자부심', '자기소개', '메구밍', '거짓말'],
        ['하루', '고생', '폭렬 마법', '내일', '응원'],
        ['매력', '홍마족', '천재', '폭렬 마법', '흑발'],
        ['홍마족', '유머', '안대', '후후', '봉인'],
        ['가치', '동료', '카즈마', '폭렬 마법', '함께'],
        ['술', '슈와슈와', '어른', '아쿠아', '카즈마'],

        ['MBTI', 'INFP', '폭렬 마법', '감정', '가치관'],
        ['아쿠아', '양배추', '양상추', '안타깝', '보상'],
        ['외로움', '폭렬 마법', '카즈마', '두렵', '든든'],
        ['다크니스', '저주', '아쿠아', '베르디아', '책임'],
        ['엘리트', '고생', '미츠루기', '파티', '의지'],
        ['카즈마', '돌아', '걱정', '위험', '표정'],
        ['다크니스', '파티', '새 동료', '카즈마', '모집'],
        ['도와', '안락소녀', '불쌍한', '마음', '조심'],
        ['융융', '혼자', '친구', '의지', '걱정'],
        ['카즈마', '실망', '진심', '솔직', '용감'],

        ['인구', '중국', '폭렬 마법', '홍마족', '나라'],
        ['스트레스', '폭렬 마법', '현대사회', '취미', '활동'],
        ['한국', '일본', '서울', '도쿄', '방문'],
        ['나라', '카즈마', '고향', '일본', '폭렬 마법'],
        ['기후 변화', '에너지', '자연', '변화', '협력'],
        ['탈세', '세금', '카즈마', '아쿠아', '모험가'],
        ['소설', '마법사', '폭렬 마법', '영웅', '마왕'],
        ['부자', '가족', '부모님', '폭렬 마법', '귀여운'],
        ['농사', '크리에이트', '마법', '땅', '물'],
        ['제목', '폭렬 마법', '홍마족', '마법사', '메구밍']
        ]

    ## 메인 함수 실행
    main(input_file, output_file, keyword)
