import whisper
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
import torch
from openai import OpenAI
import os
from dotenv import load_dotenv
import librosa
from scipy.stats import zscore

load_dotenv()

class GOPPronunciationEvaluator:
    """
    Advanced pronunciation evaluation using GOP (Goodness of Pronunciation)
    and multi-level acoustic analysis
    """

    def __init__(self):
        """Initialize with Whisper model and OpenAI client"""
        self.whisper_model = whisper.load_model("base")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def identify_speakers_simple(self, segments: List) -> Tuple[str, str]:
        """Simple speaker identification based on word count"""
        speaker_words = {}

        for segment in segments:
            # Simple energy-based speaker detection
            # Assumption: segments alternate between speakers
            speaker_id = f"Speaker{chr(65 + (segment['id'] % 2))}"  # A or B

            if speaker_id not in speaker_words:
                speaker_words[speaker_id] = 0

            text = segment.get('text', '')
            speaker_words[speaker_id] += len(text.split())

        # Student typically speaks less than teacher
        speakers = sorted(speaker_words.items(), key=lambda x: x[1])
        student_speaker = speakers[0][0] if speakers else "SpeakerA"
        teacher_speaker = speakers[1][0] if len(speakers) > 1 else "SpeakerB"

        return student_speaker, teacher_speaker

    def preview_speakers(self, audio_path: str) -> Dict[str, List[str]]:
        """Preview speaker segments to help identify who is who"""
        print("\n🎤 화자 미리보기 (Speaker Preview)...")

        # Quick transcription for preview
        result = self.whisper_model.transcribe(
            audio_path,
            language="en",
            word_timestamps=False,
            verbose=False,
            fp16=False
        )

        speaker_samples = {"SpeakerA": [], "SpeakerB": []}

        for i, segment in enumerate(result.get("segments", [])[:10]):  # First 10 segments
            speaker_id = f"Speaker{chr(65 + (i % 2))}"  # Alternate A/B
            text = segment.get('text', '').strip()
            if text:
                speaker_samples[speaker_id].append(text)

        return speaker_samples

    def extract_acoustic_features(self, audio_path: str) -> Dict:
        """Extract detailed acoustic features for pronunciation analysis"""

        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)

        # Extract features
        features = {}

        # 1. MFCC (Mel-frequency cepstral coefficients) - captures spectral characteristics
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfcc, axis=1).tolist()
        features['mfcc_std'] = np.std(mfcc, axis=1).tolist()

        # 2. Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
        features['spectral_centroid_std'] = float(np.std(spectral_centroid))

        # 3. Zero crossing rate (articulation clarity)
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))

        # 4. Energy/RMS (speaking confidence)
        rms = librosa.feature.rms(y=y)
        features['rms_mean'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))

        # 5. Pitch/F0 contour (intonation)
        f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
        f0_clean = f0[~np.isnan(f0)]
        if len(f0_clean) > 0:
            features['f0_mean'] = float(np.mean(f0_clean))
            features['f0_std'] = float(np.std(f0_clean))
            features['f0_range'] = float(np.max(f0_clean) - np.min(f0_clean))
        else:
            features['f0_mean'] = 150.0
            features['f0_std'] = 20.0
            features['f0_range'] = 50.0

        return features

    def calculate_gop_score(self, audio_path: str, reference_text: str = None,
                           speaker_filter: str = None) -> Dict:
        """
        Calculate GOP-like scores using Whisper's internal probabilities
        GOP = log P(acoustic|phoneme) - log P(acoustic|competing phonemes)
        """

        print("Calculating GOP scores...")

        # Get detailed transcription with word-level info
        result = self.whisper_model.transcribe(
            audio_path,
            language="en",
            word_timestamps=True,
            verbose=False,
            fp16=False,
            temperature=0.0,  # Deterministic for consistency
            best_of=5,  # Multiple beams for better probability estimates
            beam_size=5
        )

        # Identify speakers if needed
        if speaker_filter == "auto":
            student_speaker, teacher_speaker = self.identify_speakers_simple(result.get("segments", []))
            print(f"   Detected: Student={student_speaker}, Teacher={teacher_speaker}")
            speaker_filter = student_speaker

        # Extract word-level GOP scores
        gop_scores = []
        word_details = []
        all_segments = []

        for segment in result.get("segments", []):
            # Simple speaker assignment based on segment index
            speaker_id = f"Speaker{chr(65 + (segment['id'] % 2))}"  # A or B
            segment['speaker'] = speaker_id
            all_segments.append(segment)

            # Skip if not the target speaker
            if speaker_filter and speaker_id != speaker_filter:
                continue

            for word_data in segment.get("words", []):
                word = word_data.get("word", "").strip()

                # Whisper probability as proxy for GOP
                probability = word_data.get("probability", 0)

                # Convert to GOP-like score (lower is worse)
                # GOP typically ranges from -infinity to 0
                gop = np.log(probability + 1e-10)  # Add epsilon to avoid log(0)

                # Normalize to 0-100 scale (0=worst, 100=best)
                normalized_gop = min(100, max(0, (gop + 5) * 20))  # Empirical scaling

                gop_scores.append(normalized_gop)
                word_details.append({
                    "word": word,
                    "speaker": speaker_id,
                    "probability": probability,
                    "gop_score": normalized_gop,
                    "start": word_data.get("start"),
                    "end": word_data.get("end"),
                    "duration": word_data.get("end", 0) - word_data.get("start", 0)
                })

        # Calculate statistics
        if gop_scores:
            avg_gop = np.mean(gop_scores)
            std_gop = np.std(gop_scores)

            # Identify problematic words (GOP < 60)
            problematic_words = [w for w in word_details if w["gop_score"] < 60]

            # Convert to 1-5 scale
            pronunciation_score = 1 + (avg_gop / 100) * 4

            # Penalize high variability (inconsistent pronunciation)
            if std_gop > 20:
                pronunciation_score *= 0.9

        else:
            avg_gop = 50
            std_gop = 0
            problematic_words = []
            pronunciation_score = 3.0

        return {
            "average_gop": round(avg_gop, 2),
            "std_gop": round(std_gop, 2),
            "pronunciation_score": round(pronunciation_score, 2),
            "total_words": len(word_details),
            "problematic_words": problematic_words,
            "word_details": word_details,
            "transcript": result["text"],
            "speaker_filter": speaker_filter,
            "all_segments": all_segments
        }

    def analyze_phoneme_patterns(self, word_details: List[Dict]) -> Dict:
        """Analyze common pronunciation error patterns"""

        patterns = {
            "low_confidence_endings": [],  # Final consonant deletion
            "short_words_unclear": [],  # Articles, prepositions often unclear
            "long_words_unclear": [],  # Complex vocabulary pronunciation
            "consistent_errors": {}  # Words repeatedly mispronounced
        }

        for word in word_details:
            word_text = word["word"].lower()
            gop = word["gop_score"]

            # Check for final consonant issues
            if word_text.endswith(('t', 'd', 'k', 'p', 'b', 'g')) and gop < 70:
                patterns["low_confidence_endings"].append(word_text)

            # Short function words
            if len(word_text) <= 3 and gop < 60:
                patterns["short_words_unclear"].append(word_text)

            # Long/complex words
            if len(word_text) >= 8 and gop < 70:
                patterns["long_words_unclear"].append(word_text)

            # Track repeated errors
            if gop < 60:
                if word_text in patterns["consistent_errors"]:
                    patterns["consistent_errors"][word_text] += 1
                else:
                    patterns["consistent_errors"][word_text] = 1

        return patterns

    def calculate_fluency_score(self, word_details: List[Dict]) -> float:
        """Calculate fluency based on timing and pauses"""

        if len(word_details) < 2:
            return 3.0

        pauses = []
        speaking_rates = []

        for i in range(1, len(word_details)):
            # Calculate pause between words
            pause = word_details[i]["start"] - word_details[i-1]["end"]
            if pause > 0:
                pauses.append(pause)

            # Calculate speaking rate (words per second in segments)
            duration = word_details[i]["duration"]
            if duration > 0:
                rate = len(word_details[i]["word"].split()) / duration
                speaking_rates.append(rate)

        # Analyze pauses
        avg_pause = np.mean(pauses) if pauses else 0.2
        pause_variability = np.std(pauses) if pauses else 0.1

        # Analyze speaking rate
        avg_rate = np.mean(speaking_rates) if speaking_rates else 3.0
        rate_variability = np.std(speaking_rates) if speaking_rates else 0.5

        # Calculate fluency score (1-5)
        fluency = 5.0

        # Penalize long pauses
        if avg_pause > 0.5:
            fluency -= min(1.5, (avg_pause - 0.5) * 2)

        # Penalize high pause variability
        if pause_variability > 0.5:
            fluency -= 0.5

        # Penalize very slow or fast rate
        if avg_rate < 2.0 or avg_rate > 5.0:
            fluency -= 0.5

        # Penalize inconsistent rate
        if rate_variability > 1.0:
            fluency -= 0.5

        return max(1.0, min(5.0, fluency))

    def comprehensive_evaluation(self, audio_path: str, reference_text: str = None,
                               speaker_choice: str = None) -> Dict:
        """Perform comprehensive pronunciation evaluation"""

        print("\n⏳ Performing comprehensive pronunciation analysis...")

        # 1. GOP analysis with speaker filtering
        gop_result = self.calculate_gop_score(audio_path, reference_text, speaker_filter=speaker_choice)

        # 2. Acoustic features
        acoustic_features = self.extract_acoustic_features(audio_path)

        # 3. Phoneme pattern analysis
        patterns = self.analyze_phoneme_patterns(gop_result["word_details"])

        # 4. Fluency analysis
        fluency_score = self.calculate_fluency_score(gop_result["word_details"])

        # 5. Calculate articulation clarity from acoustic features
        zcr_score = 5.0
        if acoustic_features["zcr_mean"] < 0.05:  # Too low ZCR = mumbling
            zcr_score -= 1.0
        if acoustic_features["zcr_std"] > 0.05:  # High variability = inconsistent
            zcr_score -= 0.5
        articulation_score = max(1.0, min(5.0, zcr_score))

        # 6. Calculate intonation score from F0
        intonation_score = 5.0
        f0_range = acoustic_features.get("f0_range", 50)
        if f0_range < 30:  # Monotonous
            intonation_score -= 1.5
        elif f0_range > 200:  # Too variable
            intonation_score -= 0.5
        intonation_score = max(1.0, min(5.0, intonation_score))

        # 7. Overall pronunciation score (weighted average)
        overall_score = (
            gop_result["pronunciation_score"] * 0.4 +  # GOP weight
            fluency_score * 0.2 +
            articulation_score * 0.2 +
            intonation_score * 0.2
        )

        return {
            "overall_pronunciation_score": round(overall_score, 2),
            "component_scores": {
                "gop_pronunciation": gop_result["pronunciation_score"],
                "fluency": round(fluency_score, 2),
                "articulation_clarity": round(articulation_score, 2),
                "intonation": round(intonation_score, 2)
            },
            "gop_analysis": {
                "average_gop": gop_result["average_gop"],
                "std_gop": gop_result["std_gop"],
                "total_words": gop_result["total_words"],
                "problematic_words": gop_result["problematic_words"][:10]  # Top 10
            },
            "error_patterns": {
                "final_consonant_deletion": len(patterns["low_confidence_endings"]),
                "unclear_function_words": len(patterns["short_words_unclear"]),
                "complex_word_issues": len(patterns["long_words_unclear"]),
                "consistent_errors": patterns["consistent_errors"]
            },
            "acoustic_summary": {
                "spectral_centroid": acoustic_features["spectral_centroid_mean"],
                "zero_crossing_rate": acoustic_features["zcr_mean"],
                "pitch_range": acoustic_features.get("f0_range", 0),
                "energy_level": acoustic_features["rms_mean"]
            },
            "transcript": gop_result["transcript"],
            "evaluated_speaker": gop_result.get("speaker_filter", "All"),
            "detailed_word_scores": gop_result["word_details"][:20]  # First 20 words
        }

    def generate_feedback(self, evaluation: Dict) -> str:
        """Generate detailed feedback using GPT-4"""

        prompt = f"""Based on comprehensive pronunciation analysis, provide feedback:

Overall Score: {evaluation['overall_pronunciation_score']}/5.0

Component Scores:
- GOP Pronunciation: {evaluation['component_scores']['gop_pronunciation']}/5.0
- Fluency: {evaluation['component_scores']['fluency']}/5.0
- Articulation Clarity: {evaluation['component_scores']['articulation_clarity']}/5.0
- Intonation: {evaluation['component_scores']['intonation']}/5.0

Error Patterns:
- Final consonant deletion issues: {evaluation['error_patterns']['final_consonant_deletion']} words
- Unclear function words: {evaluation['error_patterns']['unclear_function_words']} words
- Complex word pronunciation: {evaluation['error_patterns']['complex_word_issues']} words

Problematic words (GOP < 60):
{json.dumps(evaluation['gop_analysis']['problematic_words'][:5], indent=2)}

Provide:
1. Main pronunciation strengths
2. Specific areas needing improvement
3. 2-3 practical exercises for improvement

Be constructive and specific. Format as a brief paragraph."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert pronunciation coach analyzing comprehensive acoustic data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=250
            )
            return response.choices[0].message.content
        except:
            return "Could not generate detailed feedback."

def main():
    print("="*60)
    print("    GOP-Based Pronunciation Evaluation")
    print("    (Goodness of Pronunciation + Acoustic Analysis)")
    print("="*60)

    # Input audio file
    audio_file = input("\nAudio file path (m4a/mp3/wav): ").strip() or "/Users/jason/Downloads/35526.m4a"

    if not Path(audio_file).exists():
        print(f"❌ File not found: {audio_file}")
        return

    evaluator = GOPPronunciationEvaluator()

    # Preview speakers first
    print("\n⏳ 화자 분석 중...")
    speaker_samples = evaluator.preview_speakers(audio_file)

    print("\n" + "="*60)
    print("              화자 미리보기 (Speaker Preview)")
    print("="*60)

    print("\n🔵 Speaker A 샘플:")
    for i, text in enumerate(speaker_samples["SpeakerA"][:3], 1):
        print(f"   {i}. {text[:80]}...")

    print("\n🔴 Speaker B 샘플:")
    for i, text in enumerate(speaker_samples["SpeakerB"][:3], 1):
        print(f"   {i}. {text[:80]}...")

    # Word count analysis
    words_a = sum(len(text.split()) for text in speaker_samples["SpeakerA"])
    words_b = sum(len(text.split()) for text in speaker_samples["SpeakerB"])

    print("\n📊 단어 수 분석:")
    print(f"   Speaker A: {words_a} words")
    print(f"   Speaker B: {words_b} words")

    if words_a < words_b:
        print(f"   → Speaker A가 학생일 가능성 높음 (적게 말함)")
    else:
        print(f"   → Speaker B가 학생일 가능성 높음 (적게 말함)")

    print("="*60)

    # Speaker selection with better context
    print("\n화자 선택 (Speaker Selection):")
    print("  A) Speaker A")
    print("  B) Speaker B")
    print("  auto) 자동 감지 (적게 말하는 화자를 학생으로)")
    print("  all) 전체 평가")

    speaker_input = input("\n평가할 화자 선택 (default: auto): ").strip().lower() or "auto"

    speaker_choice = None
    if speaker_input == 'a':
        speaker_choice = "SpeakerA"
        print(f"✓ Speaker A 선택됨")
    elif speaker_input == 'b':
        speaker_choice = "SpeakerB"
        print(f"✓ Speaker B 선택됨")
    elif speaker_input == 'auto':
        speaker_choice = "auto"
        print(f"✓ 자동 감지 모드")
    elif speaker_input == 'all':
        speaker_choice = None
        print(f"✓ 전체 평가 모드")

    # Optional reference text
    use_reference = input("\nHave reference text? (y/N): ").lower() == 'y'
    reference_text = ""

    if use_reference:
        reference_text = input("Enter reference text or SRT path: ").strip()
        if reference_text.endswith('.srt') and Path(reference_text).exists():
            # Extract text from SRT
            import re
            with open(reference_text, 'r') as f:
                content = f.read()
                lines = content.split('\n')
                reference_text = ' '.join([l for l in lines if l and not re.match(r'^\d+$', l)
                                         and '-->' not in l])

    # Perform evaluation
    result = evaluator.comprehensive_evaluation(audio_file, reference_text, speaker_choice)

    print("\n" + "="*60)
    print("           PRONUNCIATION EVALUATION RESULTS")
    print("="*60)

    print(f"\n👤 Evaluated Speaker: {result.get('evaluated_speaker', 'All')}")
    print(f"🎯 Overall Pronunciation Score: {result['overall_pronunciation_score']}/5.0")
    print("-" * 50)

    print("\n📊 Component Scores:")
    for component, score in result['component_scores'].items():
        print(f"   • {component.replace('_', ' ').title()}: {score}/5.0")

    print(f"\n📈 GOP Analysis:")
    print(f"   • Average GOP: {result['gop_analysis']['average_gop']}/100")
    print(f"   • Standard Deviation: {result['gop_analysis']['std_gop']}")
    print(f"   • Total Words: {result['gop_analysis']['total_words']}")

    if result['gop_analysis']['problematic_words']:
        print(f"\n🔴 Problematic Words (GOP < 60):")
        for word in result['gop_analysis']['problematic_words'][:5]:
            print(f"   • '{word['word']}' (GOP: {word['gop_score']:.1f})")

    print(f"\n🔍 Error Patterns Detected:")
    patterns = result['error_patterns']
    if patterns['final_consonant_deletion'] > 0:
        print(f"   • Final consonant deletion: {patterns['final_consonant_deletion']} instances")
    if patterns['unclear_function_words'] > 0:
        print(f"   • Unclear function words: {patterns['unclear_function_words']} instances")
    if patterns['complex_word_issues'] > 0:
        print(f"   • Complex word pronunciation: {patterns['complex_word_issues']} instances")

    if patterns['consistent_errors']:
        print(f"\n   Consistently mispronounced:")
        for word, count in list(patterns['consistent_errors'].items())[:3]:
            print(f"      • '{word}' ({count} times)")

    # Generate AI feedback
    print("\n💬 Detailed Feedback:")
    feedback = evaluator.generate_feedback(result)
    print(f"   {feedback}")

    # Save results
    output_file = audio_file.rsplit('.', 1)[0] + '_gop_evaluation.json'
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print(f"💾 Full results saved to: {output_file}")
    print("="*60)

if __name__ == "__main__":
    main()