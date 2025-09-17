# AssemblyAI Speech Evaluation Setup

## 🚀 AssemblyAI vs Whisper+Pyannote 비교

| 기능 | Whisper+Pyannote | AssemblyAI |
|------|-----------------|------------|
| **화자 분리** | Pyannote (2-3분) | 내장 (30초-1분) |
| **발음 평가** | GOP 근사치 | 신뢰도 점수 |
| **추가 기능** | 없음 | 요약, 하이라이트, 감정분석 |
| **설정 복잡도** | 복잡 (HF Token 필요) | 간단 (API Key만) |
| **처리 속도** | 2-3분 | 30초-1분 |
| **오프라인** | 가능 | 불가 (API 호출) |
| **비용** | 무료 | 유료 ($0.37/hr) |

## 🔑 API Key 설정

1. AssemblyAI 계정 생성: https://www.assemblyai.com/
2. API Key 발급
3. `.env` 파일에 추가:

```bash
# .env 파일
ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here
OPENAI_API_KEY=sk-your-openai-key-here
```

## 📦 설치

```bash
# 가상환경 활성화
source .venv/bin/activate

# 새 의존성 설치
pip install requests>=2.31.0

# 또는 전체 requirements 재설치
pip install -r requirements.txt
```

## 🎯 AssemblyAI 고급 기능

### 1. **자동 화자 분리**
```python
"speaker_labels": True  # 자동으로 Speaker A, B, C... 구분
```

### 2. **신뢰도 기반 발음 평가**
```python
"confidence_threshold": 0.1  # 낮은 신뢰도 단어도 포함
# 0.6 미만: very_unclear
# 0.75 미만: unclear
```

### 3. **AI 인사이트**
- **Auto Highlights**: 중요 구문 자동 추출
- **Summarization**: 대화 요약
- **Auto Chapters**: 주제별 구분
- **Sentiment Analysis**: 감정 분석
- **Entity Detection**: 인명, 장소 등 인식

### 4. **발화 장애 감지**
```python
"disfluencies": True  # "um", "uh", "like" 등 감지
```

## 🚀 사용법

```bash
python speech_test_ass.py
```

### 입력 예시:
```
Audio file path: /Users/jason/Downloads/35526.m4a
Course level 1-4: 2
Student speaker ID: (auto-detect)
```

## 📊 결과 해석

### 종합 점수:
- **Pronunciation Score**: AssemblyAI 신뢰도 기반 (1-5점)
- **Content Score**: GPT-4 내용 평가 (1-5점)
- **Combined Score**: 가중평균 (발음 30% + 내용 70%)

### AI 인사이트:
- **Key Phrases**: "English learning", "pronunciation practice"
- **Summary**: "Student discusses language learning experience"
- **Sentiment**: Positive/Neutral/Negative
- **Entities**: 언급된 인명, 장소, 조직

## 💡 장점

1. **빠른 처리**: 2-3분 → 30초-1분
2. **간편 설정**: HuggingFace Token 불필요
3. **고급 기능**: 요약, 감정분석, 주제 추출
4. **정확한 화자 분리**: 산업용 수준
5. **클라우드 기반**: 로컬 GPU 불필요

## ⚠️ 단점

1. **유료 서비스**: $0.37/시간 (약 450원/시간)
2. **인터넷 필수**: 오프라인 불가
3. **데이터 전송**: 오디오 파일 업로드 필요

## 🔄 비교 테스트

같은 오디오 파일로 두 방법 비교:

```bash
# 기존 방법 (2-3분)
python speech_test2.py

# AssemblyAI 방법 (30초-1분)
python speech_test_ass.py
```

## 💰 비용 계산

- **월 10시간 평가**: $3.70 (약 4,500원)
- **월 50시간 평가**: $18.50 (약 22,000원)
- **일 1시간씩 30일**: $11.10 (약 13,300원)

교육용으로는 충분히 합리적인 가격!