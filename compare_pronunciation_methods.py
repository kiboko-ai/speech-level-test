"""
Comparison of Different Pronunciation Evaluation Methods

1. pronunciation_eval.py: Whisper Confidence Scores (Basic)
2. gop_pronunciation_eval.py: GOP + Acoustic Analysis (Advanced)
3. phoneme_align_eval.py: Phoneme-level Alignment (Most Detailed)
"""

import sys
from pathlib import Path
from typing import Dict

def compare_methods():
    """Compare different pronunciation evaluation methods"""

    print("="*70)
    print("         발음 평가 방법 비교 (Pronunciation Evaluation Comparison)")
    print("="*70)

    methods = {
        "1": {
            "name": "Whisper Confidence Scores",
            "file": "pronunciation_eval.py",
            "pros": [
                "빠른 처리 속도 (30-40초)",
                "간단한 구현",
                "전체적인 명확도 평가 가능",
                "GPU 없이도 실행 가능"
            ],
            "cons": [
                "세부적인 발음 오류 파악 어려움",
                "음소 단위 분석 불가",
                "한국인 특유 오류 패턴 감지 제한적"
            ],
            "accuracy": "70-75%",
            "speed": "Fast",
            "use_case": "빠른 전체 평가가 필요할 때"
        },
        "2": {
            "name": "GOP + Acoustic Analysis",
            "file": "gop_pronunciation_eval.py",
            "pros": [
                "업계 표준 GOP 점수 사용",
                "음향 특징 분석 (pitch, energy, spectral)",
                "유창성과 억양 평가 포함",
                "오류 패턴 자동 감지"
            ],
            "cons": [
                "처리 시간이 조금 더 걸림 (1-2분)",
                "더 많은 계산 리소스 필요",
                "Librosa 등 추가 라이브러리 필요"
            ],
            "accuracy": "85-90%",
            "speed": "Medium",
            "use_case": "정확한 종합 평가가 필요할 때"
        },
        "3": {
            "name": "Phoneme-level Alignment (Montreal Forced Aligner)",
            "file": "phoneme_align_eval.py (미구현)",
            "pros": [
                "음소 단위의 정확한 분석",
                "L1 interference 패턴 정확히 감지",
                "각 음소의 정확한 타이밍 정보",
                "언어학적으로 가장 정확한 방법"
            ],
            "cons": [
                "설치와 설정이 복잡함",
                "처리 시간이 오래 걸림 (2-3분+)",
                "음향 모델 훈련 필요할 수 있음",
                "전문 지식 필요"
            ],
            "accuracy": "95%+",
            "speed": "Slow",
            "use_case": "연구나 전문적인 평가가 필요할 때"
        }
    }

    print("\n📊 방법별 비교:\n")

    for key, method in methods.items():
        print(f"\n{key}. {method['name']} ({method['file']})")
        print("-" * 60)

        print("  ✅ 장점:")
        for pro in method['pros']:
            print(f"     • {pro}")

        print("  ⚠️ 단점:")
        for con in method['cons']:
            print(f"     • {con}")

        print(f"\n  📈 정확도: {method['accuracy']}")
        print(f"  ⏱️ 속도: {method['speed']}")
        print(f"  🎯 사용 시나리오: {method['use_case']}")

    print("\n" + "="*70)
    print("                        권장 사항")
    print("="*70)

    recommendations = """
1. 🚀 빠른 스크리닝이 필요한 경우:
   → pronunciation_eval.py (Whisper Confidence)
   → 많은 학생을 빠르게 평가할 때 적합

2. ⚖️ 균형잡힌 평가가 필요한 경우:
   → gop_pronunciation_eval.py (GOP + Acoustic)
   → 정확도와 속도의 균형, 실제 교육 현장에 최적

3. 🔬 정밀 분석이 필요한 경우:
   → phoneme_align_eval.py (Forced Alignment)
   → 연구 목적이나 심화 교정이 필요한 경우

4. 💡 하이브리드 접근법 (추천):
   → 1차: Whisper로 빠른 스크리닝
   → 2차: 문제가 있는 학생만 GOP 상세 평가
   → 3차: 특별 관리가 필요한 경우 Phoneme 분석
"""

    print(recommendations)

    print("\n📌 한국 학습자를 위한 추가 고려사항:")
    korean_considerations = [
        "• /r/ vs /l/ 구분 (rice/lice)",
        "• /f/ vs /p/ 구분 (coffee/copy)",
        "• /v/ vs /b/ 구분 (very/berry)",
        "• /θ/ vs /s/ 구분 (think/sink)",
        "• /z/ vs /j/ 구분 (zip/jip)",
        "• 종성 자음 탈락 (last → las)",
        "• 모음 길이 구분 (ship/sheep)",
        "• 강세와 리듬 패턴"
    ]

    for consideration in korean_considerations:
        print(consideration)

    print("\n" + "="*70)
    print("\n실행 가능한 스크립트:")
    print("  1. python pronunciation_eval.py")
    print("  2. python gop_pronunciation_eval.py")
    print("\n선택하여 실행하세요!")
    print("="*70)

if __name__ == "__main__":
    compare_methods()