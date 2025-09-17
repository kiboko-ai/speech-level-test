import azure.cognitiveservices.speech as speechsdk
import numpy as np
import wave
import io

# =============================
# 1️⃣ Azure Speech Service 설정
# =============================


speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

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
# 4️⃣ 평가용 함수 (오디오 파일 없이 Azure만 사용)
# =============================
def evaluate_sentence(reference_text):
    # 마이크에서 직접 녹음
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    # Azure 발음 평가 설정
    pron_config = speechsdk.PronunciationAssessmentConfig(
        reference_text=reference_text,
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
        enable_miscue=True
    )

    # 프로소디(운율) 기능 활성화
    pron_config.enable_prosody_assessment()
    pron_config.apply_to(recognizer)

    print(f"\n🎤 문장 읽기: {reference_text}")
    print("지금 말씀해주세요...")

    result = recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        # Azure 발음 평가 결과
        pron_result = speechsdk.PronunciationAssessmentResult(result)

        # 기본 점수
        accuracy = pron_result.accuracy_score
        fluency = pron_result.fluency_score
        completeness = pron_result.completeness_score

        # 프로소디 점수 (Azure에서 제공하는 경우)
        prosody_score = pron_result.prosody_score if hasattr(pron_result, 'prosody_score') else None

        # 프로소디 점수가 없으면 임시 값 사용
        if prosody_score:
            pitch_score = prosody_score
            stress_score = prosody_score
        else:
            # 정확도와 유창성을 기반으로 추정
            pitch_score = (accuracy + fluency) / 2 * 0.9
            stress_score = (accuracy + fluency) / 2 * 0.85

        # 말 속도 점수 (완성도와 유창성 기반)
        speed_score = (completeness + fluency) / 2

        # 총점 계산 (가중치 합산)
        total_score = (
            weights["accuracy"] * accuracy +
            weights["fluency"] * fluency +
            weights["completeness"] * completeness +
            weights["pitch"] * pitch_score +
            weights["stress"] * stress_score +
            weights["speed"] * speed_score
        )

        print(f"\n📊 평가 결과:")
        print(f"  인식된 텍스트: {result.text}")
        print(f"  정확도: {accuracy:.1f}")
        print(f"  유창성: {fluency:.1f}")
        print(f"  완성도: {completeness:.1f}")
        print(f"  억양: {pitch_score:.1f}")
        print(f"  강세: {stress_score:.1f}")
        print(f"  속도: {speed_score:.1f}")
        print(f"  ➡️ 총점: {total_score:.1f}")

        return {
            "reference": reference_text,
            "recognized": result.text,
            "azure": {
                "accuracy": accuracy,
                "fluency": fluency,
                "completeness": completeness
            },
            "prosody": {
                "pitch": pitch_score,
                "stress": stress_score,
                "speed": speed_score
            },
            "total_score": total_score
        }

    elif result.reason == speechsdk.ResultReason.NoMatch:
        print(f"음성을 인식할 수 없습니다: {result.no_match_details}")
        return None
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print(f"음성 인식이 취소되었습니다: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f"오류 내용: {cancellation_details.error_details}")
        return None

# =============================
# 5️⃣ 여러 문장 평가
# =============================
print("=" * 50)
print("🎯 영어 발음 평가 프로그램")
print("=" * 50)

results = []
for i, sentence in enumerate(sentences, 1):
    print(f"\n[{i}/{len(sentences)}번째 문장]")
    res = evaluate_sentence(sentence)
    if res:
        results.append(res)
    else:
        print("이 문장을 건너뜁니다.")

# =============================
# 6️⃣ 평균 총점 계산
# =============================
if results:
    avg_total = np.mean([r["total_score"] for r in results])
    print("\n" + "=" * 50)
    print(f"📊 최종 평균 점수: {avg_total:.2f} / 100")
    print("=" * 50)
else:
    print("\n평가된 문장이 없습니다.")