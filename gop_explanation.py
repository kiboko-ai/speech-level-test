"""
GOP 동작 원리 시연 - Reference 유무 차이
"""

import whisper
import numpy as np

def demonstrate_gop_difference():
    """Reference text 유무에 따른 GOP 차이 설명"""

    print("="*60)
    print("   GOP: Reference Text 유무 차이")
    print("="*60)

    # 가상의 Whisper 출력 시뮬레이션

    # Case 1: Reference 없이 (현재 구현)
    print("\n📌 Case 1: Reference Text 없음")
    print("-" * 40)

    # Whisper가 여러 후보를 검토
    candidates = {
        "I like rice": 0.4,
        "I like lice": 0.55,  # 최종 선택
        "I like ice": 0.05
    }

    print("Whisper 후보 분석:")
    for text, prob in candidates.items():
        gop = np.log(prob + 1e-10)
        normalized = min(100, max(0, (gop + 5) * 20))
        selected = "← 선택됨" if prob == 0.55 else ""
        print(f"  '{text}': 확률={prob:.2f}, GOP={normalized:.1f} {selected}")

    print("\n결과: 'I like lice'로 전사")
    print("→ GOP = 63.0 (중간 정도 확신도)")
    print("→ 발음이 불명확하다는 것은 알 수 있음")
    print("→ 하지만 'rice'를 'lice'로 잘못 발음했는지는 모름")

    # Case 2: Reference 있을 때 (이상적)
    print("\n\n📌 Case 2: Reference Text 있음")
    print("-" * 40)
    print("Reference: 'I like rice'")
    print("학습자 발화: [음향 신호]")

    print("\n전통적 GOP 계산:")
    print("  1. Forced Alignment:")
    print("     'rice' → [r][aɪ][s] 음소 정렬")

    print("\n  2. 각 음소별 GOP:")
    print("     [r] 위치의 음향:")
    print("       P(음향|r) = 0.3  (낮음)")
    print("       P(음향|l) = 0.7  (높음)")
    print("       → GOP[r] = log(0.3) - log(0.7) = -0.85")
    print("       → 'r'을 'l'로 발음함 감지!")

    print("\n     [aɪ] 위치의 음향:")
    print("       P(음향|aɪ) = 0.9 (높음)")
    print("       → GOP[aɪ] = 0.95 (정확)")

    print("\n     [s] 위치의 음향:")
    print("       P(음향|s) = 0.95 (높음)")
    print("       → GOP[s] = 0.98 (정확)")

    print("\n결과: 전체 GOP = 낮음")
    print("→ 구체적으로 'r'→'l' 오류 감지 가능")
    print("→ 정확한 교정 피드백 제공 가능")

    # 요약
    print("\n\n" + "="*60)
    print("                    요약")
    print("="*60)

    summary = """
🔍 Reference Text 없을 때 (현재 gop_pronunciation_eval.py):
   • Whisper의 confidence를 GOP 대용으로 사용
   • "얼마나 명확하게 발음했는가?" 측정
   • 전반적인 발음 품질 평가
   • 구체적인 음소 오류는 감지 못함

🎯 Reference Text 있을 때 (이상적):
   • 정확한 음소 단위 비교
   • "무엇을 잘못 발음했는가?" 감지
   • r/l, p/f 등 구체적 오류 파악
   • 맞춤형 교정 피드백 가능

💡 현실적 해결책:
   1. 자유 대화: Reference 없이 평가 (명확도만)
   2. 읽기 과제: Reference 활용 (정확도+명확도)
   3. SRT 파일: 부분적 Reference로 활용
"""
    print(summary)

    print("\n현재 구현의 한계:")
    print("• 'rice'를 'lice'로 발음해도 Whisper가 'lice'로")
    print("  정확히 인식하면 GOP가 높게 나옴")
    print("• 즉, 잘못된 발음이지만 '명확한' 경우 감지 못함")

    print("\n그래도 유용한 이유:")
    print("• 머뭇거림, 불명확한 발음 감지 가능")
    print("• 전반적인 유창성 평가 가능")
    print("• 자신감 없는 발화 구간 파악 가능")

if __name__ == "__main__":
    demonstrate_gop_difference()