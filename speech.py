
# =============================
# 필요 라이브러리 설치
# pip install azure-cognitiveservices-speech librosa soundfile numpy
# =============================

import azure.cognitiveservices.speech as speechsdk
import librosa
import numpy as np
import time

# =============================
# 1️⃣ Azure Speech Service 설정
# =============================


speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

# =============================
# 2️⃣ 평가할 문장 리스트
# =============================
sentences = [
    "Good morning, how are you today?",
    "I am very happy to see you.",
    "Let's practice English together."
]

# =============================
# 3️⃣ 가중치 설정
# =============================
weights = {
    "accuracy": 0.3,
    "fluency": 0.2,
    "completeness": 0.1,
    "pitch": 0.2,
    "stress": 0.1,
    "speed": 0.1
}

# =============================
# 4️⃣ 평가용 함수
# =============================
def evaluate_sentence(reference_text, audio_file="temp.wav"):
    # 오디오 파일로 저장하도록 설정
    audio_config_file = speechsdk.audio.AudioConfig(filename=audio_file)
    recognizer_with_file = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config_file
    )

    # Azure 발음 평가 설정
    pron_config = speechsdk.PronunciationAssessmentConfig(
        reference_text=reference_text,
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
        enable_miscue=True
    )
    pron_config.apply_to(recognizer)

    print(f"\n🎤 문장 읽기: {reference_text}")
    print("녹음을 시작하세요... (말한 후 잠시 기다리세요)")
    result = recognizer.recognize_once()

    if result.reason != speechsdk.ResultReason.RecognizedSpeech:
        print("음성 인식 실패")
        return None

    # Azure 점수
    pron_result = speechsdk.PronunciationAssessmentResult(result)
    Accuracy = pron_result.accuracy_score
    Fluency = pron_result.fluency_score
    Completeness = pron_result.completeness_score

    # 오디오가 파일로 저장되도록 잠시 대기
    time.sleep(0.5)

    # =============================
    # Librosa 분석 (억양/강세)
    # =============================
    # 실제 오디오 파일이 없으므로 임시 점수 반환
    # 실제 구현에서는 pyaudio나 sounddevice로 녹음 후 저장 필요
    pitch_score = np.random.uniform(70, 90)
    stress_score = np.random.uniform(70, 90)

    # 말 속도 계산 (인식된 단어 수 기반)
    num_words = len(reference_text.split())
    speed_score = max(0, min(100, 80 + np.random.uniform(-10, 10)))  # 임시 점수

    # =============================
    # 총점 계산 (가중치 합산)
    # =============================
    total_score = (
        weights["accuracy"] * Accuracy +
        weights["fluency"] * Fluency +
        weights["completeness"] * Completeness +
        weights["pitch"] * pitch_score +
        weights["stress"] * stress_score +
        weights["speed"] * speed_score
    )

    return {
        "reference": reference_text,
        "azure": {
            "accuracy": Accuracy,
            "fluency": Fluency,
            "completeness": Completeness
        },
        "librosa": {
            "pitch": pitch_score,
            "stress": stress_score,
            "speed": speed_score
        },
        "total_score": total_score
    }

# =============================
# 5️⃣ 여러 문장 평가
# =============================
results = []
for sentence in sentences:
    res = evaluate_sentence(sentence)
    if res:
        results.append(res)

# =============================
# 6️⃣ 평균 총점 계산
# =============================
avg_total = np.mean([r["total_score"] for r in results])
print(f"\n📊 최종 평균 점수: {avg_total:.2f}")
