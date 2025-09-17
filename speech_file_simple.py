import librosa
import numpy as np

# 오디오 파일
audio_file = "/Users/jason/Downloads/35526.m4a"

# =============================
# 가중치 설정
# =============================
weights = {
    "pitch": 0.3,
    "stress": 0.3,
    "speed": 0.2,
    "energy": 0.2
}

# =============================
# 오디오 파일 평가
# =============================
print(f"🎤 오디오 파일 분석 중: {audio_file}")

try:
    # 오디오 로드
    y, sr = librosa.load(audio_file, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    print(f"⏱️ 파일 길이: {duration:.2f}초")

    # 1. Pitch (피치/억양) 분석
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=75, fmax=400)
    f0_clean = f0[~np.isnan(f0)]  # NaN 제거

    if len(f0_clean) > 0:
        pitch_variation = np.std(f0_clean)
        pitch_score = max(0, min(100, 100 - pitch_variation/2))  # 변동이 적당해야 높은 점수
    else:
        pitch_score = 50

    # 2. Stress (강세) - RMS 에너지
    rms = librosa.feature.rms(y=y)[0]
    stress_variation = np.std(rms) / (np.mean(rms) + 1e-6)  # 변동 계수
    stress_score = max(0, min(100, 80 + stress_variation * 100))  # 적절한 강세 변화

    # 3. Speed (속도) - Zero Crossing Rate로 대체 측정
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    speech_rate = np.mean(zcr) * 1000  # 정규화
    speed_score = max(0, min(100, 100 - abs(speech_rate - 50)))  # 50이 이상적

    # 4. Energy (에너지) - 전체적인 음성 에너지
    energy = np.mean(rms)
    energy_score = max(0, min(100, energy * 500))  # 정규화

    # 총점 계산
    total_score = (
        weights["pitch"] * pitch_score +
        weights["stress"] * stress_score +
        weights["speed"] * speed_score +
        weights["energy"] * energy_score
    )

    # 결과 출력
    print("\n📊 평가 결과:")
    print(f"  🎵 억양(Pitch): {pitch_score:.1f}/100")
    print(f"  💪 강세(Stress): {stress_score:.1f}/100")
    print(f"  ⏱️ 속도(Speed): {speed_score:.1f}/100")
    print(f"  🔊 에너지(Energy): {energy_score:.1f}/100")
    print(f"\n✨ 총점: {total_score:.1f}/100")

    # 레벨 판정
    if total_score >= 80:
        level = "Excellent"
    elif total_score >= 70:
        level = "Good"
    elif total_score >= 60:
        level = "Average"
    else:
        level = "Needs Improvement"

    print(f"📈 레벨: {level}")

except FileNotFoundError:
    print(f"❌ 파일을 찾을 수 없습니다: {audio_file}")
except Exception as e:
    print(f"❌ 오류 발생: {e}")