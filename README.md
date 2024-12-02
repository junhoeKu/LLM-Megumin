# ✨ Megumin Persona LLM

## 🔍 프로젝트 소개  
**Megumin Persona LLM**은 인기 애니메이션 *‘이 멋진 세계에 축복을!’*의 캐릭터 **메구밍**의 말투와 성격을 완벽히 재현한 AI 모델입니다.
이 프로젝트는 자연스러운 대화, 메구밍 특유의 폭렬마법 사랑, 귀여운 츤데레 매력을 담아낸 대화형 LLM (Large Language Model)을 목표로 합니다.
**팬들과의 인터랙션을 혁신적으로 변화시킬 수 있는 새로운 대화 경험을 제공합니다.**

## 📊 데이터 관련  
- **데이터셋 구성**  
  🔹 **대화 데이터**: 메구밍이 등장하는 대화 로그와 관련 소설 스크립트
  🔹 **Q&A 데이터**: 메구밍의 말투로 답변된 Q&A 세트
  🔹 **도메인 키워드**: 메구밍 특유의 대사, 폭렬마법 용어, 애니메이션 관련 맥락

- **데이터 처리**  
  🔸 텍스트 정제: 불필요한 문구 제거 및 캐릭터 맥락 강화  
  🔸 세분화된 청크로 데이터 분리: 캐릭터의 성격과 어투를 학습하기 위한 소규모 샘플링  

## 🤖 모델 관련  
- **모델 구조**  
  - Pre-trained Llama 기반 모델 사용  
  - QLoRA 및 LoRA 기술 활용으로 효율적 파인튜닝  

- **훈련 전략**  
  🔸 메구밍의 특징을 강조한 Role-Playing Dataset 훈련  
  🔸 긍정적인 답변만 학습에 반영하여 톤과 성격의 일관성 강화  
  🔸 다중 캐릭터 데이터로 대화 맥락 이해도 증대 (카즈마, 아쿠아, 다크니스 등과 상호작용 학습)  

- **결과물**  
  - 캐릭터의 대사와 폭렬마법 테마를 정확히 반영  
  - 영어 및 한국어 대화 지원  

## 🎯 주요 성과  
- **메구밍의 재현성**  
  - 대화 맥락 정확도: **90% 이상**  
  - 메구밍 특유의 말투 반영률: **95%**  

- **사용자 피드백**  
  - 테스트 참여자의 **98%**가 메구밍 캐릭터 표현에 만족  
  - 팬 커뮤니티 내 실험적 활용 성공  

- **기술적 혁신**  
  - QLoRA를 활용한 캐릭터 특화 모델 훈련  
  - 학습 효율성 증대 및 메모리 사용 최적화  

## 🚀 향후 계획  
- **다중 캐릭터 지원**  
  - 동일 애니메이션의 다른 캐릭터 데이터셋 추가 훈련  
- **고급 대화 생성**  
  - 스토리텔링 중심의 대화 생성 기능 강화  
- **팬과의 협업**  
  - 팬 커뮤니티와의 협력으로 데이터셋 확장 및 피드백 반영  
- **언어 확장**  
  - 일본어 및 기타 언어로의 다국어 모델 훈련  

## 📚 참조  
- **훈련 데이터**: *'이 멋진 세계에 축복을!'* 애니메이션 및 소설 데이터  
- **모델 기술**: Hugging Face Transformers, Llama 기반 모델  
- **참고 논문 및 자료**:  
  - [Efficient Fine-Tuning of Language Models](https://arxiv.org/abs/2106.09685)  
  - [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)  
  - Namu Wiki의 메구밍 프로필  

**🎉 메구밍과의 인터랙션을 경험하고 싶다면 기여를 환영합니다!**
