# English Level Test Speech Evaluation System

## 시스템 개요
이 시스템은 학생-교사 영어 대화 음성 파일을 분석하여 학생의 영어 수준을 평가합니다.

## 주요 기능

### 1. **음성 인식 및 화자 분리**
- OpenAI Whisper를 사용한 정확한 영어 음성 인식
- 화자 분리를 통한 학생/교사 구분
- SRT 형식의 타임스탬프 포함 스크립트 생성

### 2. **평가 영역 (Rubric)**

#### Content Relevance (내용 관련성)
- Task Coverage: 과제 수행 완성도
- Appropriateness: 주제 적절성

#### Accuracy (정확성)
- Grammar Control: 문법 정확도
- Vocabulary Use: 어휘 사용 범위와 정확성

#### Coherence (일관성)
- Logical Flow: 논리적 흐름
- Cohesive Devices: 연결어 사용

#### Delivery (전달력)
- Pronunciation: 발음 정확도
- Intonation & Stress: 억양과 강세

### 3. **발음 오류 분석**
한국인 학습자의 일반적인 발음 오류 패턴 감지:
- **r/l 혼동**: right↔light, problem↔ploblem
- **p/f 혼동**: coffee↔copy, office↔oppice
- **b/v 혼동**: very↔berry, video↔bideo
- **th 발음**: think↔sink, with↔wis
- **z/s 혼동**: zero↔sero, amazing↔amasing
- **끝자음 탈락**: want↔wan, friend↔frien

### 4. **점수 체계**
- 각 항목 1.0~5.0점 (0.5점 단위)
- 레벨 1-2: 기초 의사소통, 기본 어휘 중심
- 레벨 3-4: 정확성, 복잡한 구조, 유창성 중심

## 설치 방법

```bash
# 1. 의존성 설치
./setup.sh

# 2. 환경 변수 설정
export OPENAI_API_KEY='your-openai-api-key'
export HUGGINGFACE_TOKEN='your-huggingface-token'  # 선택사항
```

## 사용 방법

```bash
# 기본 실행 (audio_file 경로는 코드에 하드코딩됨)
python speech_test2.py

# 코드 수정하여 다른 파일 분석
# speech_test2.py의 main() 함수에서 audio_file 변수 수정
```

## 출력 결과

### 콘솔 출력
- 각 평가 영역별 점수
- 평균 점수
- 사용된 주요 어휘/표현
- 강점과 약점 피드백
- 발음 오류 분석

### JSON 파일 저장
- `{audio_filename}_evaluation.json`
- 모든 평가 데이터 포함
- 상세 발음 분석 데이터

## 평가 프로세스

1. **음성 인식**: Whisper로 전체 대화 transcribe
2. **화자 분리**: 학생과 교사 구분
3. **GPT-4 평가**: Content, Accuracy, Coherence 평가
4. **발음 분석**: 오인식 패턴으로 발음 문제 감지
5. **억양 분석**: Librosa로 pitch 변화 분석
6. **종합 점수**: 모든 영역 통합하여 최종 점수 산출

## 주의사항

- OPENAI_API_KEY 필수 (GPT-4 평가용)
- HUGGINGFACE_TOKEN 권장 (정확한 화자 분리)
- 음성 파일은 명확한 음질 권장
- 학생-교사 1:1 대화 형식에 최적화됨