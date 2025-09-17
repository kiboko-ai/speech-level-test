import requests
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class AssemblyAIAdvancedEvaluator:
    """
    Advanced AssemblyAI Speech Evaluation with:
    - Universal-1 model (최신/가장 똑똑함)
    - LeMUR AI reasoning engine
    - Advanced analysis features
    """

    def __init__(self):
        """Initialize AssemblyAI and OpenAI clients"""
        self.api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not self.api_key:
            raise ValueError("ASSEMBLYAI_API_KEY not found")

        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.base_url = "https://api.assemblyai.com/v2"
        self.headers = {"authorization": self.api_key}

    def upload_file(self, file_path: str) -> str:
        """Upload audio file to AssemblyAI"""
        print("📤 Uploading to AssemblyAI (Advanced)...")

        with open(file_path, 'rb') as f:
            response = requests.post(
                f"{self.base_url}/upload",
                files={'file': f},
                headers=self.headers
            )

        if response.status_code == 200:
            upload_url = response.json()['upload_url']
            print("✅ Upload successful")
            return upload_url
        else:
            raise Exception(f"Upload failed: {response.text}")

    def transcribe_with_universal_model(self, upload_url: str, model_type: str = None) -> str:
        """Transcribe using advanced model"""
        if model_type:
            print(f"🧠 Using {model_type} model...")
        else:
            print("🧠 Using default AssemblyAI model...")

        # Advanced configuration
        config = {
            "audio_url": upload_url,

            # Core features
            "speaker_labels": True,
            "speakers_expected": 2,  # Hint: teacher + student

            # Language and quality
            "language_detection": True,  # Auto-detect language
            "punctuate": True,
            "format_text": True,
            "disfluencies": True,  # Capture "um", "ah"

            # Advanced analysis
            "sentiment_analysis": True,
            "entity_detection": True,
            "auto_highlights": True,

            # Quality settings
            "word_boost": ["pronunciation", "fluency", "grammar", "vocabulary"],
        }

        # Add speech model only if specified
        if model_type:
            config["speech_model"] = model_type

        # Submit transcription
        response = requests.post(
            f"{self.base_url}/transcript",
            json=config,
            headers=self.headers
        )

        if response.status_code == 200:
            transcript_id = response.json()['id']
            print(f"✅ Advanced transcription job: {transcript_id}")
            return transcript_id
        else:
            raise Exception(f"Advanced transcription failed: {response.text}")

    def poll_transcription(self, transcript_id: str) -> Dict[str, Any]:
        """Poll for transcription completion with progress"""
        print("⏳ Processing with advanced AI models...")

        start_time = time.time()
        while True:
            response = requests.get(
                f"{self.base_url}/transcript/{transcript_id}",
                headers=self.headers
            )

            if response.status_code == 200:
                result = response.json()
                status = result['status']
                elapsed = int(time.time() - start_time)

                if status == 'completed':
                    print(f"✅ Advanced analysis completed! ({elapsed}s)")
                    return result
                elif status == 'error':
                    raise Exception(f"Analysis error: {result.get('error', 'Unknown')}")
                else:
                    print(f"⏳ {status.title()}... ({elapsed}s)")
                    time.sleep(3)
            else:
                raise Exception(f"Polling failed: {response.text}")

    def use_lemur_analysis(self, transcript_id: str, student_text: str, course_level: int) -> Dict:
        """Use LeMUR (AI reasoning engine) for advanced analysis"""
        print("🤖 Analyzing with LeMUR AI reasoning engine...")

        # Advanced prompt for educational analysis
        lemur_prompt = f"""You are an expert English language evaluator analyzing student speech.

Course Level: {course_level}/4
Student Text: {student_text}

Perform comprehensive analysis:

1. PRONUNCIATION ASSESSMENT:
   - Identify specific pronunciation errors
   - Assess clarity and intelligibility
   - Rate fluency and rhythm (1-5)

2. LANGUAGE PROFICIENCY:
   - Grammar accuracy and complexity
   - Vocabulary range and appropriateness
   - Sentence structure variety

3. COMMUNICATION SKILLS:
   - Coherence and organization
   - Task completion effectiveness
   - Interactive communication ability

4. SPECIFIC FEEDBACK:
   - Strengths to reinforce
   - Priority areas for improvement
   - Concrete practice suggestions

5. LEVEL ASSESSMENT:
   - Current proficiency estimate
   - Readiness for next level
   - Specific skill gaps

Provide detailed, actionable feedback in JSON format with numerical scores (1-5) and specific examples."""

        # Submit LeMUR request
        lemur_config = {
            "transcript_ids": [transcript_id],
            "prompt": lemur_prompt,
            "final_model": "default",  # Could use "anthropic/claude-3-5-sonnet" if available
            "max_output_size": 2000,
            "temperature": 0.1  # Low temperature for consistent evaluation
        }

        try:
            response = requests.post(
                f"{self.base_url}/lemur/v3/generate/summary",
                json=lemur_config,
                headers=self.headers
            )

            if response.status_code == 200:
                lemur_result = response.json()
                print("✅ LeMUR analysis completed!")

                # Try to parse as JSON, fallback to text
                try:
                    analysis = json.loads(lemur_result['response'])
                    return analysis
                except:
                    return {
                        "lemur_analysis": lemur_result['response'],
                        "analysis_type": "text_format"
                    }
            else:
                print(f"⚠️ LeMUR unavailable: {response.text}")
                return {"lemur_analysis": "LeMUR analysis unavailable"}

        except Exception as e:
            print(f"⚠️ LeMUR error: {e}")
            return {"lemur_analysis": f"LeMUR error: {str(e)}"}

    def identify_student_by_context(self, utterances: List[Dict]) -> str:
        """Identify student speaker using conversation context and patterns"""

        if not utterances:
            return "A"

        # Get all speakers
        speakers = list(set(u['speaker'] for u in utterances))
        if len(speakers) == 1:
            return speakers[0]

        print("🔍 Analyzing conversation context to identify student...")

        # Analyze each speaker's characteristics
        speaker_analysis = {}

        for speaker in speakers:
            speaker_utterances = [u for u in utterances if u['speaker'] == speaker]
            all_text = " ".join([u['text'] for u in speaker_utterances])

            # Calculate metrics for each speaker
            analysis = {
                'total_words': len(all_text.split()),
                'utterance_count': len(speaker_utterances),
                'avg_utterance_length': len(all_text.split()) / len(speaker_utterances),
                'sample_texts': [u['text'][:100] for u in speaker_utterances[:3]]
            }

            # Teacher indicators (질문, 지시, 피드백 패턴)
            teacher_patterns = [
                # Questions
                r'\b(what|where|when|why|how|can you|could you|would you|do you|did you|will you|are you|is it)\b',
                # Instructions
                r'\b(tell me|explain|describe|talk about|let\'s|now|next|first|then|finally)\b',
                # Feedback
                r'\b(good|great|excellent|nice|well done|correct|right|yes|okay|I see|interesting)\b',
                # Teacher language
                r'\b(lesson|homework|assignment|practice|study|learn|understand|remember)\b',
                # Corrections
                r'\b(actually|no|not quite|try again|listen|repeat|pronunciation)\b'
            ]

            # Student indicators (답변, 불확실성, 학습자 언어)
            student_patterns = [
                # Uncertainty
                r'\b(I think|maybe|perhaps|I\'m not sure|I don\'t know|um|uh|well)\b',
                # Learning language
                r'\b(I learned|I studied|I practice|I try|I want to|I like|I don\'t understand)\b',
                # Response patterns
                r'\b(yes|no|okay|alright|sure|of course)\s*[.!]',
                # Personal references
                r'\b(I|me|my|mine|myself)\b',
                # Hesitation markers
                r'\b(um|uh|er|ah|well|so|like)\b'
            ]

            import re

            # Count pattern matches
            teacher_score = 0
            student_score = 0

            for pattern in teacher_patterns:
                teacher_score += len(re.findall(pattern, all_text.lower()))

            for pattern in student_patterns:
                student_score += len(re.findall(pattern, all_text.lower()))

            # Conversation position analysis
            first_speaker_bonus = 5 if speaker == utterances[0]['speaker'] else 0
            teacher_score += first_speaker_bonus  # Teachers often start conversations

            # Question vs answer pattern
            questions = len(re.findall(r'\?', all_text))
            if questions > 0:
                teacher_score += questions * 3

            # Length-based scoring (teachers often give longer explanations)
            if analysis['avg_utterance_length'] > 15:
                teacher_score += 2
            elif analysis['avg_utterance_length'] < 8:
                student_score += 2

            analysis.update({
                'teacher_score': teacher_score,
                'student_score': student_score,
                'likely_role': 'teacher' if teacher_score > student_score else 'student',
                'confidence': abs(teacher_score - student_score)
            })

            speaker_analysis[speaker] = analysis

        # Display analysis for debugging
        print("\n📊 Speaker Analysis Results:")
        for speaker, data in speaker_analysis.items():
            role = data['likely_role']
            conf = data['confidence']
            words = data['total_words']
            print(f"   Speaker {speaker}: {role.title()} (confidence: {conf}, words: {words})")
            print(f"      Sample: \"{data['sample_texts'][0][:60]}...\"")

        # Find student (highest student_score or lowest teacher_score)
        student_candidates = [(s, data) for s, data in speaker_analysis.items()
                            if data['likely_role'] == 'student']

        if student_candidates:
            # Choose student with highest confidence
            student_speaker = max(student_candidates, key=lambda x: x[1]['confidence'])[0]
        else:
            # Fallback: choose speaker with lowest teacher score
            student_speaker = min(speaker_analysis.keys(),
                                key=lambda s: speaker_analysis[s]['teacher_score'])

        print(f"🎯 Final decision: Speaker {student_speaker} identified as student")
        return student_speaker

    def advanced_pronunciation_analysis(self, result: Dict, target_speaker: str) -> Dict:
        """Enhanced pronunciation analysis with Universal-1 data"""

        words = result.get('words', [])
        utterances = result.get('utterances', [])

        # Filter student words
        student_words = []
        student_utterances = [u for u in utterances if u['speaker'] == target_speaker]

        for utterance in student_utterances:
            start_time = utterance['start']
            end_time = utterance['end']

            # Find words in this utterance timeframe
            utterance_words = [
                w for w in words
                if start_time <= w['start'] <= end_time
            ]
            student_words.extend(utterance_words)

        if not student_words:
            return {
                "pronunciation_score": 3.0,
                "detailed_analysis": "No student words detected",
                "total_words": 0
            }

        # Advanced confidence analysis
        confidence_scores = [w.get('confidence', 0.8) for w in student_words]

        # Categorize by confidence levels
        very_clear = [w for w in student_words if w.get('confidence', 0) > 0.9]
        clear = [w for w in student_words if 0.8 <= w.get('confidence', 0) <= 0.9]
        unclear = [w for w in student_words if 0.6 <= w.get('confidence', 0) < 0.8]
        very_unclear = [w for w in student_words if w.get('confidence', 0) < 0.6]

        # Calculate advanced metrics
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        clarity_ratio = len(very_clear + clear) / len(student_words)

        # Pronunciation score calculation
        base_score = 1 + (avg_confidence * 4)

        # Adjustments based on distribution
        if clarity_ratio > 0.8:
            base_score += 0.3
        elif clarity_ratio < 0.5:
            base_score -= 0.5

        pronunciation_score = max(1.0, min(5.0, base_score))

        return {
            "pronunciation_score": round(pronunciation_score, 2),
            "average_confidence": round(avg_confidence, 3),
            "total_words": len(student_words),
            "clarity_distribution": {
                "very_clear": len(very_clear),
                "clear": len(clear),
                "unclear": len(unclear),
                "very_unclear": len(very_unclear)
            },
            "clarity_ratio": round(clarity_ratio, 3),
            "problematic_words": [
                {
                    "word": w['text'],
                    "confidence": w.get('confidence', 0),
                    "timestamp": w['start'] / 1000
                }
                for w in very_unclear[:10]
            ]
        }

    def comprehensive_evaluation(self, audio_path: str, course_level: int = 2,
                               model_type: str = "universal-1", use_lemur: bool = True) -> Dict:
        """Perform comprehensive evaluation with advanced models"""

        print("🚀 Starting ADVANCED AssemblyAI evaluation...")
        print(f"🧠 Model selected: {model_type if model_type else 'Default (auto)'}")
        print(f"🤖 LeMUR analysis: {'Enabled' if use_lemur else 'Disabled'}")

        try:
            # Step 1: Upload
            upload_url = self.upload_file(audio_path)

            # Step 2: Advanced transcription
            transcript_id = self.transcribe_with_universal_model(upload_url, model_type)

            # Step 3: Wait for completion
            result = self.poll_transcription(transcript_id)

            # Step 4: Identify student speaker using context analysis
            utterances = result.get('utterances', [])
            if not utterances:
                raise Exception("No speaker diarization results")

            student_speaker = self.identify_student_by_context(utterances)
            print(f"👤 Student identified: Speaker {student_speaker}")

            # Step 5: Extract student content and calculate speaker distribution
            student_text = ""
            speaker_words = {}

            for utterance in utterances:
                speaker = utterance['speaker']
                words = len(utterance['text'].split())
                speaker_words[speaker] = speaker_words.get(speaker, 0) + words

                if utterance['speaker'] == student_speaker:
                    student_text += utterance['text'] + " "

            # Step 6: Advanced pronunciation analysis
            pronunciation_analysis = self.advanced_pronunciation_analysis(result, student_speaker)

            # Step 7: LeMUR AI analysis (optional)
            lemur_analysis = {}
            if use_lemur and student_text.strip():
                lemur_analysis = self.use_lemur_analysis(transcript_id, student_text.strip(), course_level)

            # Step 8: Enhanced insights
            insights = {
                "highlights": (result.get('auto_highlights_result', {}).get('results') or [])[:5],
                "sentiment": result.get('sentiment_analysis_results') or [],
                "entities": (result.get('entities') or [])[:5],
                "language_confidence": result.get('language_confidence', 0),
                "detected_language": result.get('language_code', 'en'),
                "disfluencies_detected": bool(result.get('disfluencies')),
                "audio_duration": result.get('audio_duration', 0) / 1000  # Convert to seconds
            }

            # Step 9: Compile advanced results
            return {
                "audio_file": audio_path,
                "model_used": model_type if model_type else "default",
                "course_level": course_level,
                "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),

                # Speaker analysis
                "student_speaker": student_speaker,
                "speaker_distribution": speaker_words,
                "student_word_count": len(student_text.split()),

                # Transcription
                "full_transcript": result.get('text', ''),
                "student_text": student_text.strip(),
                "confidence_score": result.get('confidence', 0),

                # Advanced pronunciation
                "pronunciation_analysis": pronunciation_analysis,

                # AI insights
                "assemblyai_insights": insights,

                # LeMUR analysis
                "lemur_analysis": lemur_analysis,

                # Performance metrics
                "processing_time": result.get('audio_duration', 0) / 1000,
                "model_performance": {
                    "overall_confidence": result.get('confidence', 0),
                    "language_detection_confidence": result.get('language_confidence', 0),
                    "speaker_separation_quality": len(utterances) / max(1, len(set(speaker_words.keys())))
                }
            }

        except Exception as e:
            print(f"❌ Advanced evaluation error: {e}")
            raise e

def main():
    print("="*70)
    print("     AssemblyAI ADVANCED Speech Evaluation")
    print("        Advanced Models + LeMUR AI Engine")
    print("="*70)

    # Input parameters
    audio_file = input("\nAudio file path: ").strip() or "/Users/jason/Downloads/35526.mp3"

    if not Path(audio_file).exists():
        print(f"❌ File not found: {audio_file}")
        return

    course_level = int(input("Course level 1-4 (default: 2): ").strip() or "2")

    print("\nSelect AI model:")
    print("  1) universal (최신, 가장 똑똑함) - 추천")
    print("  2) best (높은 정확도)")
    print("  3) nano (빠름)")
    print("  4) default (기본 - 자동 선택)")

    model_choice = input("Model choice (default: 4): ").strip() or "4"

    # Model mapping with correct names
    model_map = {
        "1": "universal",  # Changed from universal-1
        "2": "best",
        "3": "nano",
        "4": None,  # None means let AssemblyAI choose default
        "nano": "nano",  # Support direct model name input
        "best": "best",
        "universal": "universal"
    }
    model_type = model_map.get(model_choice, None)

    use_lemur = input("\nUse LeMUR AI reasoning engine? (Y/n): ").strip().lower() != 'n'

    try:
        # Initialize advanced evaluator
        evaluator = AssemblyAIAdvancedEvaluator()

        # Perform advanced evaluation
        results = evaluator.comprehensive_evaluation(
            audio_file, course_level, model_type, use_lemur
        )

        # Display results
        print("\n" + "="*70)
        print("                 ADVANCED EVALUATION RESULTS")
        print("="*70)

        print(f"\n🧠 Model Used: {results.get('model_used', 'Default') if results.get('model_used') else 'Default (auto)'}")
        print(f"👤 Student: Speaker {results['student_speaker']}")
        print(f"💬 Student Words: {results['student_word_count']}")
        print(f"⏱️ Audio Duration: {results['assemblyai_insights']['audio_duration']:.1f}s")

        # Pronunciation analysis
        pronunciation = results["pronunciation_analysis"]
        print(f"\n🎤 Advanced Pronunciation Analysis:")
        print(f"   Score: {pronunciation['pronunciation_score']}/5.0")
        print(f"   Average Confidence: {pronunciation['average_confidence']:.1%}")
        print(f"   Clarity Ratio: {pronunciation['clarity_ratio']:.1%}")

        clarity = pronunciation['clarity_distribution']
        print(f"\n   Word Clarity Distribution:")
        print(f"   • Very Clear: {clarity['very_clear']} words")
        print(f"   • Clear: {clarity['clear']} words")
        print(f"   • Unclear: {clarity['unclear']} words")
        print(f"   • Very Unclear: {clarity['very_unclear']} words")

        # LeMUR analysis
        if results.get('lemur_analysis') and 'lemur_analysis' not in str(results['lemur_analysis']):
            print(f"\n🤖 LeMUR AI Analysis:")
            lemur = results['lemur_analysis']
            if isinstance(lemur, dict):
                for key, value in lemur.items():
                    if isinstance(value, (int, float)):
                        print(f"   {key}: {value}")
                    elif len(str(value)) < 100:
                        print(f"   {key}: {value}")

        # Insights
        insights = results["assemblyai_insights"]
        if insights['highlights']:
            print(f"\n🔍 Key Phrases:")
            for highlight in insights['highlights']:
                print(f"   • {highlight.get('text', 'N/A')}")

        # Save results
        output_file = Path(audio_file).stem + "_advanced_assemblyai.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n" + "="*70)
        print(f"💾 Advanced results saved to: {output_file}")
        print("="*70)

    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()