import whisper
import numpy as np
from pathlib import Path
import json
from typing import Dict, List
import torch
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class PronunciationEvaluator:
    def __init__(self):
        """Initialize with Whisper model"""
        self.whisper_model = whisper.load_model("base")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def evaluate_pronunciation_with_confidence(self, audio_path: str, reference_srt: str = None):
        """Evaluate pronunciation using Whisper confidence scores"""

        print("Transcribing with confidence scores...")

        # Get transcription with detailed info
        result = self.whisper_model.transcribe(
            audio_path,
            language="en",
            word_timestamps=True,
            verbose=False,
            fp16=False  # For better accuracy
        )

        # Analyze confidence scores
        pronunciation_issues = []
        total_words = 0
        low_confidence_words = []

        for segment in result.get("segments", []):
            for word_data in segment.get("words", []):
                total_words += 1
                word = word_data.get("word", "").strip()
                confidence = word_data.get("probability", 0)  # Whisper uses "probability"

                # Categorize by confidence
                if confidence < 0.5:
                    category = "very_unclear"
                    low_confidence_words.append({
                        "word": word,
                        "confidence": confidence,
                        "category": category,
                        "start": word_data.get("start"),
                        "end": word_data.get("end")
                    })
                elif confidence < 0.7:
                    category = "unclear"
                    low_confidence_words.append({
                        "word": word,
                        "confidence": confidence,
                        "category": category,
                        "start": word_data.get("start"),
                        "end": word_data.get("end")
                    })
                elif confidence < 0.85:
                    category = "somewhat_unclear"
                    # Only track if it's a content word
                    if len(word) > 3:  # Skip short function words
                        low_confidence_words.append({
                            "word": word,
                            "confidence": confidence,
                            "category": category,
                            "start": word_data.get("start"),
                            "end": word_data.get("end")
                        })

        # Calculate pronunciation score
        if total_words > 0:
            avg_confidence = sum(w.get("probability", 0) for s in result["segments"]
                               for w in s.get("words", [])) / total_words

            # Convert to 1-5 scale
            pronunciation_score = 1 + (avg_confidence * 4)  # Maps 0-1 to 1-5

            # Adjust based on severe issues
            very_unclear_count = len([w for w in low_confidence_words if w["category"] == "very_unclear"])
            if very_unclear_count > total_words * 0.1:  # >10% very unclear
                pronunciation_score = min(pronunciation_score, 2.5)

        else:
            pronunciation_score = 3.0
            avg_confidence = 0

        return {
            "pronunciation_score": round(pronunciation_score, 1),
            "average_confidence": round(avg_confidence, 3),
            "total_words": total_words,
            "unclear_words": low_confidence_words,
            "unclear_count": len(low_confidence_words),
            "transcript": result["text"],
            "detailed_segments": result["segments"]
        }

    def compare_with_reference(self, transcribed: str, reference: str):
        """Compare transcribed text with reference SRT"""
        from difflib import SequenceMatcher

        # Clean and normalize texts
        trans_words = transcribed.lower().split()
        ref_words = reference.lower().split()

        # Find differences
        matcher = SequenceMatcher(None, ref_words, trans_words)
        differences = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                differences.append({
                    "type": "substitution",
                    "reference": ' '.join(ref_words[i1:i2]),
                    "transcribed": ' '.join(trans_words[j1:j2]),
                    "likely_issue": "pronunciation"
                })
            elif tag == 'delete':
                differences.append({
                    "type": "deletion",
                    "reference": ' '.join(ref_words[i1:i2]),
                    "transcribed": "",
                    "likely_issue": "word_dropped"
                })
            elif tag == 'insert':
                differences.append({
                    "type": "insertion",
                    "reference": "",
                    "transcribed": ' '.join(trans_words[j1:j2]),
                    "likely_issue": "extra_sound"
                })

        # Calculate accuracy
        accuracy = matcher.ratio()

        return {
            "accuracy": round(accuracy, 3),
            "differences": differences,
            "difference_count": len(differences)
        }

    def get_pronunciation_feedback(self, evaluation_result: Dict) -> str:
        """Generate pronunciation feedback using GPT-4"""

        unclear_words = evaluation_result.get("unclear_words", [])

        if not unclear_words:
            return "Pronunciation is generally clear with no significant issues detected."

        prompt = f"""Based on automatic speech recognition confidence scores, analyze these pronunciation issues:

Unclear words (low confidence):
{json.dumps(unclear_words[:20], indent=2)}  # Top 20

Total words: {evaluation_result['total_words']}
Average confidence: {evaluation_result['average_confidence']}
Pronunciation score: {evaluation_result['pronunciation_score']}/5.0

Provide:
1. Main pronunciation patterns observed
2. Specific sounds that need work
3. 2-3 practical improvement tips

Be constructive and specific. Format as a brief paragraph."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a pronunciation coach analyzing ASR confidence data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=200
            )
            return response.choices[0].message.content
        except:
            return "Could not generate detailed feedback."

def main():
    print("="*60)
    print("    Pronunciation Evaluation (Confidence-Based)")
    print("="*60)

    # Input audio file
    audio_file = input("\nAudio file path (m4a/mp3/wav): ").strip() or "/Users/jason/Downloads/35526.m4a"

    if not Path(audio_file).exists():
        print(f"❌ File not found: {audio_file}")
        return

    # Optional: reference SRT
    use_reference = input("Have reference SRT for comparison? (y/N): ").lower() == 'y'
    reference_text = ""

    if use_reference:
        srt_file = input("SRT file path: ").strip()
        if Path(srt_file).exists():
            with open(srt_file, 'r') as f:
                # Extract just the text from SRT
                import re
                content = f.read()
                lines = content.split('\n')
                reference_text = ' '.join([l for l in lines if l and not re.match(r'^\d+$', l)
                                         and '-->' not in l and not l.startswith('[')
                                         and not l.startswith('<')])

    evaluator = PronunciationEvaluator()

    print("\n⏳ Analyzing pronunciation...")
    result = evaluator.evaluate_pronunciation_with_confidence(audio_file)

    print("\n" + "="*60)
    print("              PRONUNCIATION ANALYSIS")
    print("="*60)

    print(f"\n📊 Overall Metrics:")
    print(f"   • Pronunciation Score: {result['pronunciation_score']}/5.0")
    print(f"   • Average Confidence: {result['average_confidence']:.1%}")
    print(f"   • Total Words: {result['total_words']}")
    print(f"   • Unclear Words: {result['unclear_count']}")

    # Show unclear words by category
    unclear = result['unclear_words']
    if unclear:
        print("\n🔴 Pronunciation Issues:")

        very_unclear = [w for w in unclear if w['category'] == 'very_unclear']
        unclear_only = [w for w in unclear if w['category'] == 'unclear']
        somewhat = [w for w in unclear if w['category'] == 'somewhat_unclear']

        if very_unclear:
            print(f"\n   Very Unclear ({len(very_unclear)} words):")
            for w in very_unclear[:5]:
                print(f"      • '{w['word']}' (confidence: {w['confidence']:.1%})")
            if len(very_unclear) > 5:
                print(f"      ... and {len(very_unclear)-5} more")

        if unclear_only:
            print(f"\n   Unclear ({len(unclear_only)} words):")
            for w in unclear_only[:5]:
                print(f"      • '{w['word']}' (confidence: {w['confidence']:.1%})")
            if len(unclear_only) > 5:
                print(f"      ... and {len(unclear_only)-5} more")

    # Compare with reference if available
    if use_reference and reference_text:
        print("\n📝 Comparison with Reference:")
        comparison = evaluator.compare_with_reference(result['transcript'], reference_text)
        print(f"   • Accuracy: {comparison['accuracy']:.1%}")
        print(f"   • Differences: {comparison['difference_count']}")

        if comparison['differences'][:5]:
            print("\n   Main Differences:")
            for diff in comparison['differences'][:5]:
                if diff['type'] == 'substitution':
                    print(f"      • '{diff['reference']}' → '{diff['transcribed']}'")

    # Generate feedback
    print("\n💬 Pronunciation Feedback:")
    feedback = evaluator.get_pronunciation_feedback(result)
    print(f"   {feedback}")

    # Save results
    output_file = audio_file.rsplit('.', 1)[0] + '_pronunciation.json'
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print(f"💾 Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()