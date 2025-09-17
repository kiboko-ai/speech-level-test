import requests
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import os
from dotenv import load_dotenv
from openai import OpenAI
from difflib import SequenceMatcher

load_dotenv()

class AssemblyAIEvaluator:
    """
    English speech evaluation using AssemblyAI's advanced features:
    - Speaker Diarization
    - Confidence Scores
    - Word-level Timestamps
    - Auto Chapters (topic detection)
    """

    def __init__(self):
        """Initialize AssemblyAI and OpenAI clients"""
        self.api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not self.api_key:
            raise ValueError("ASSEMBLYAI_API_KEY not found in environment variables")

        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.base_url = "https://api.assemblyai.com/v2"
        self.headers = {"authorization": self.api_key}

    def upload_file(self, file_path: str) -> str:
        """Upload audio file to AssemblyAI"""
        print("📤 Uploading audio file to AssemblyAI...")

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

    def transcribe_audio(self, upload_url: str, enable_features: bool = True,
                        feature_priority: str = "highlights") -> str:
        """Submit transcription request with advanced features"""
        print("🎯 Starting transcription with advanced features...")

        # Start with basic configuration that definitely works
        if not enable_features:
            # Minimal config for faster processing
            config = {
                "audio_url": upload_url,
                "speaker_labels": True,
                "punctuate": True
            }
        else:
            # Full config with features
            config = {
                "audio_url": upload_url,
                "speaker_labels": True,
                "punctuate": True,
                "format_text": True,
                "auto_highlights": True
            }

        # Submit transcription job
        response = requests.post(
            f"{self.base_url}/transcript",
            json=config,
            headers=self.headers
        )

        if response.status_code == 200:
            transcript_id = response.json()['id']
            print(f"✅ Job submitted: {transcript_id}")
            return transcript_id
        else:
            raise Exception(f"Transcription failed: {response.text}")

    def poll_transcription(self, transcript_id: str) -> Dict[str, Any]:
        """Poll for transcription completion"""
        print("⏳ Waiting for transcription completion...")

        while True:
            response = requests.get(
                f"{self.base_url}/transcript/{transcript_id}",
                headers=self.headers
            )

            if response.status_code == 200:
                result = response.json()
                status = result['status']

                if status == 'completed':
                    print("✅ Transcription completed!")
                    return result
                elif status == 'error':
                    raise Exception(f"Transcription error: {result.get('error', 'Unknown error')}")
                else:
                    print(f"⏳ Status: {status}, waiting...")
                    time.sleep(3)
            else:
                raise Exception(f"Polling failed: {response.text}")

    def analyze_speakers(self, result: Dict) -> Dict[str, Any]:
        """Analyze speaker information and identify student vs teacher using context"""

        utterances = result.get('utterances', [])
        if not utterances:
            return {"student_speaker": "A", "teacher_speaker": "B", "speaker_stats": {}}

        # Use context-based identification
        student_speaker = self.identify_student_by_context(utterances)

        # Count words per speaker for stats
        speaker_stats = {}
        for utterance in utterances:
            speaker = utterance['speaker']
            words = len(utterance['text'].split())

            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    'words': 0,
                    'utterances': 0,
                    'total_confidence': 0,
                    'sample_text': []
                }

            speaker_stats[speaker]['words'] += words
            speaker_stats[speaker]['utterances'] += 1
            speaker_stats[speaker]['total_confidence'] += utterance.get('confidence', 0.8)

            # Collect sample text
            if len(speaker_stats[speaker]['sample_text']) < 3:
                speaker_stats[speaker]['sample_text'].append(utterance['text'][:80])

        # Calculate averages
        for speaker in speaker_stats:
            stats = speaker_stats[speaker]
            stats['avg_confidence'] = stats['total_confidence'] / stats['utterances']
            stats['avg_words_per_utterance'] = stats['words'] / stats['utterances']

        # Determine teacher speaker
        all_speakers = list(speaker_stats.keys())
        teacher_speaker = next((s for s in all_speakers if s != student_speaker), "B")

        return {
            "student_speaker": student_speaker,
            "teacher_speaker": teacher_speaker,
            "speaker_stats": speaker_stats,
            "total_speakers": len(all_speakers)
        }

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

            # Teacher indicators (질문, 지시, 피드백 패턴)
            teacher_patterns = [
                r'\b(what|where|when|why|how|can you|could you|would you|do you|did you|will you|are you|is it)\b',
                r'\b(tell me|explain|describe|talk about|let\'s|now|next|first|then|finally)\b',
                r'\b(good|great|excellent|nice|well done|correct|right|yes|okay|I see|interesting)\b',
                r'\b(lesson|homework|assignment|practice|study|learn|understand|remember)\b',
                r'\b(actually|no|not quite|try again|listen|repeat|pronunciation)\b'
            ]

            # Student indicators (답변, 불확실성, 학습자 언어)
            student_patterns = [
                r'\b(I think|maybe|perhaps|I\'m not sure|I don\'t know|um|uh|well)\b',
                r'\b(I learned|I studied|I practice|I try|I want to|I like|I don\'t understand)\b',
                r'\b(yes|no|okay|alright|sure|of course)\s*[.!]',
                r'\b(I|me|my|mine|myself)\b',
                r'\b(um|uh|er|ah|well|so|like)\b'
            ]

            import re

            # Count pattern matches
            teacher_score = sum(len(re.findall(pattern, all_text.lower())) for pattern in teacher_patterns)
            student_score = sum(len(re.findall(pattern, all_text.lower())) for pattern in student_patterns)

            # Additional scoring
            if speaker == utterances[0]['speaker']:
                teacher_score += 5  # Teachers often start conversations

            questions = len(re.findall(r'\?', all_text))
            teacher_score += questions * 3

            speaker_analysis[speaker] = {
                'teacher_score': teacher_score,
                'student_score': student_score,
                'likely_role': 'teacher' if teacher_score > student_score else 'student',
                'words': len(all_text.split())
            }

        # Display analysis
        print("📊 Speaker Analysis:")
        for speaker, data in speaker_analysis.items():
            role = data['likely_role']
            print(f"   Speaker {speaker}: {role.title()} (T:{data['teacher_score']}, S:{data['student_score']})")

        # Find student
        student_candidates = [s for s, data in speaker_analysis.items() if data['likely_role'] == 'student']

        if student_candidates:
            student_speaker = student_candidates[0]
        else:
            # Fallback to lowest teacher score
            student_speaker = min(speaker_analysis.keys(), key=lambda s: speaker_analysis[s]['teacher_score'])

        print(f"🎯 Student identified: Speaker {student_speaker}")
        return student_speaker

    def extract_pronunciation_issues(self, result: Dict, target_speaker: str) -> Dict:
        """Extract pronunciation issues from AssemblyAI confidence scores"""

        words = result.get('words', [])
        if not words:
            return {"pronunciation_score": 3.0, "issues": [], "total_words": 0}

        # Filter words by speaker
        speaker_words = []
        for word in words:
            # AssemblyAI words don't have speaker info directly
            # Need to match with utterances
            word_start = word['start']

            # Find corresponding utterance
            for utterance in result.get('utterances', []):
                if (utterance['start'] <= word_start <= utterance['end'] and
                    utterance['speaker'] == target_speaker):
                    speaker_words.append(word)
                    break

        if not speaker_words:
            return {"pronunciation_score": 3.0, "issues": [], "total_words": 0}

        # Analyze confidence scores
        low_confidence_words = []
        confidence_scores = []

        for word in speaker_words:
            confidence = word.get('confidence', 0.8)
            confidence_scores.append(confidence)

            # Categories based on confidence
            if confidence < 0.6:
                category = "very_unclear"
                low_confidence_words.append({
                    "word": word['text'],
                    "confidence": confidence,
                    "category": category,
                    "start": word['start'] / 1000,  # Convert to seconds
                    "end": word['end'] / 1000
                })
            elif confidence < 0.75:
                category = "unclear"
                low_confidence_words.append({
                    "word": word['text'],
                    "confidence": confidence,
                    "category": category,
                    "start": word['start'] / 1000,
                    "end": word['end'] / 1000
                })

        # Calculate pronunciation score
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            pronunciation_score = 1 + (avg_confidence * 4)  # Map to 1-5 scale

            # Penalize many unclear words
            unclear_ratio = len(low_confidence_words) / len(speaker_words)
            if unclear_ratio > 0.15:  # >15% unclear
                pronunciation_score *= 0.9

        else:
            avg_confidence = 0.8
            pronunciation_score = 3.0

        return {
            "pronunciation_score": round(pronunciation_score, 2),
            "average_confidence": round(avg_confidence, 3),
            "total_words": len(speaker_words),
            "unclear_words": low_confidence_words,
            "unclear_count": len(low_confidence_words)
        }

    def evaluate_content_with_gpt4(self, student_text: str, full_transcript: str,
                                 course_level: int = 2, highlights: List = None) -> Dict:
        """Evaluate content using GPT-4 with AssemblyAI insights"""

        # Include highlights if available
        highlights_text = ""
        if highlights:
            key_phrases = [h.get('text', '') for h in highlights[:5]]
            highlights_text = f"\nKey phrases detected: {', '.join(key_phrases)}"

        prompt = f"""You are an expert English speaking evaluator using AI transcription analysis.

Student's speech:
{student_text}

Full conversation context:
{full_transcript}
{highlights_text}

Course level: {course_level} (1-4 scale)
- Levels 1-2: Basic communication, vocabulary building
- Levels 3-4: Accuracy, complex structures, fluency

Evaluate these aspects (1.0-5.0, increment 0.1):

1. Content Relevance
   - task_coverage: Addressing topics/questions appropriately
   - appropriateness: Staying on-topic and contextually appropriate

2. Language Accuracy
   - grammar_control: Sentence structure, verb tenses, syntax
   - vocabulary_use: Range, precision, and sophistication

3. Coherence
   - logical_flow: Clear reasoning and idea development
   - cohesive_devices: Effective use of linking words/phrases

4. Engagement
   - interaction_quality: Natural conversation flow
   - response_depth: Elaboration and detail in responses

Provide analysis focusing on:
- Vocabulary sophistication for the level
- Grammar patterns (correct/incorrect)
- Conversation skills and engagement
- Specific strengths and improvement areas

Output JSON format:
{{
  "course_level": {course_level},
  "task_coverage": <score>,
  "appropriateness": <score>,
  "grammar_control": <score>,
  "vocabulary_use": <score>,
  "logical_flow": <score>,
  "cohesive_devices": <score>,
  "interaction_quality": <score>,
  "response_depth": <score>,
  "average_score": <calculated average>,
  "strengths": ["strength1", "strength2", "strength3"],
  "weaknesses": ["weakness1", "weakness2"],
  "vocabulary_level": "<basic/intermediate/advanced>",
  "grammar_patterns": ["pattern1", "pattern2"],
  "suggestions": ["suggestion1", "suggestion2", "suggestion3"],
  "detailed_feedback": "<comprehensive paragraph>"
}}"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert English evaluator. Be precise with scores and constructive with feedback."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=1000
            )
            return json.loads(response.choices[0].message.content)

        except Exception as e:
            print(f"GPT-4 evaluation error: {e}")
            return self._create_fallback_evaluation(course_level)

    def _create_fallback_evaluation(self, course_level: int) -> Dict:
        """Create fallback evaluation if GPT-4 fails"""
        return {
            "course_level": course_level,
            "task_coverage": 3.0,
            "appropriateness": 3.0,
            "grammar_control": 3.0,
            "vocabulary_use": 3.0,
            "logical_flow": 3.0,
            "cohesive_devices": 3.0,
            "interaction_quality": 3.0,
            "response_depth": 3.0,
            "average_score": 3.0,
            "strengths": ["Speech was captured and processed"],
            "weaknesses": ["Detailed analysis unavailable"],
            "vocabulary_level": "intermediate",
            "grammar_patterns": ["Mixed patterns observed"],
            "suggestions": ["Continue practicing English conversation"],
            "detailed_feedback": "Technical evaluation completed. Consider retrying for detailed feedback."
        }

    def comprehensive_evaluation(self, audio_path: str, course_level: int = 2,
                               student_speaker: str = None) -> Dict:
        """Perform comprehensive speech evaluation using AssemblyAI"""

        print("🎯 Starting comprehensive evaluation with AssemblyAI...")

        try:
            # Step 1: Upload file
            upload_url = self.upload_file(audio_path)

            # Step 2: Transcribe with features (using highlights priority)
            transcript_id = self.transcribe_audio(upload_url, enable_features=True, feature_priority="highlights")

            # Step 3: Wait for completion
            result = self.poll_transcription(transcript_id)

            # Debug: Print what we got from AssemblyAI
            print(f"🔍 Debug - Result keys: {list(result.keys()) if result else 'None'}")
            print(f"🔍 Debug - Has utterances: {'utterances' in result if result else 'No result'}")
            print(f"🔍 Debug - Has words: {'words' in result if result else 'No result'}")

            # Step 4: Analyze speakers
            speaker_analysis = self.analyze_speakers(result)

            if not student_speaker or student_speaker == "auto-detect":
                student_speaker = speaker_analysis["student_speaker"]

            print(f"👤 Student identified as: Speaker {student_speaker}")

            # Debug: Check if we have utterances
            utterances = result.get('utterances', [])
            if not utterances:
                print("⚠️ No utterances found, trying to extract from segments...")
                # Fallback: create utterances from text if speaker diarization failed
                if result.get('text'):
                    utterances = [{
                        'speaker': 'A',
                        'text': result['text'],
                        'start': 0,
                        'end': 1000,
                        'confidence': 0.8
                    }]
                    student_speaker = 'A'

            # Step 5: Extract student text
            student_text = ""
            for utterance in utterances:
                if utterance['speaker'] == student_speaker:
                    student_text += utterance['text'] + " "

            # Fallback if no student text found
            if not student_text.strip():
                print("⚠️ No student text found, using full transcript...")
                student_text = result.get('text', 'No text available')

            # Step 6: Pronunciation analysis
            pronunciation_analysis = self.extract_pronunciation_issues(result, student_speaker)

            # Step 7: Content evaluation with GPT-4
            highlights = result.get('auto_highlights_result', {}).get('results', [])
            content_evaluation = self.evaluate_content_with_gpt4(
                student_text.strip(),
                result.get('text', ''),
                course_level,
                highlights
            )

            # Step 8: Compile comprehensive results
            return {
                "audio_file": audio_path,
                "course_level": course_level,
                "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),

                # Speaker information
                "speaker_analysis": speaker_analysis,
                "evaluated_speaker": student_speaker,
                "student_word_count": len(student_text.split()),

                # Transcription
                "full_transcript": result.get('text', ''),
                "student_text": student_text.strip(),
                "utterances": utterances,

                # Pronunciation assessment
                "pronunciation": pronunciation_analysis,

                # Content evaluation
                "content_evaluation": content_evaluation,

                # AssemblyAI insights
                "insights": {
                    "highlights": highlights[:5] if highlights else [],  # Top 5 key phrases
                    "summary": result.get('summary', ''),
                    "chapters": (result.get('chapters') or [])[:3],  # Top 3 topics
                    "sentiment": result.get('sentiment_analysis_results') or [],
                    "entities": (result.get('entities') or [])[:5]  # Top 5 entities
                },

                # Overall scores
                "overall_assessment": {
                    "pronunciation_score": pronunciation_analysis["pronunciation_score"],
                    "content_score": content_evaluation["average_score"],
                    "combined_score": round(
                        (pronunciation_analysis["pronunciation_score"] * 0.3 +
                         content_evaluation["average_score"] * 0.7), 2
                    )
                }
            }

        except Exception as e:
            print(f"🔍 Debug - Error details: {str(e)}")
            print(f"🔍 Debug - Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            raise e

def main():
    print("="*70)
    print("        AssemblyAI-based English Speech Evaluation")
    print("                 Advanced AI Features")
    print("="*70)

    # Input validation
    audio_file = input("\nAudio file path (m4a/mp3/wav): ").strip()
    if not audio_file:
        audio_file = "/Users/jason/Downloads/35526.m4a"

    if not Path(audio_file).exists():
        print(f"❌ File not found: {audio_file}")
        return

    # Course level
    course_input = input("Course level 1-4 (default: 2): ").strip()
    course_level = int(course_input) if course_input.isdigit() and 1 <= int(course_input) <= 4 else 2

    # Student speaker (optional)
    speaker_input = input("Specific student speaker ID (or auto-detect): ").strip()
    student_speaker = speaker_input if speaker_input else None

    try:
        # Initialize evaluator
        evaluator = AssemblyAIEvaluator()

        # Perform evaluation
        results = evaluator.comprehensive_evaluation(audio_file, course_level, student_speaker)

        # Display results
        print("\n" + "="*70)
        print("                    EVALUATION RESULTS")
        print("="*70)

        print(f"\n📊 Overall Assessment:")
        overall = results["overall_assessment"]
        print(f"   🗣️ Pronunciation Score: {overall['pronunciation_score']}/5.0")
        print(f"   📝 Content Score: {overall['content_score']}/5.0")
        print(f"   🎯 Combined Score: {overall['combined_score']}/5.0")

        print(f"\n👤 Speaker Analysis:")
        print(f"   Student: Speaker {results['evaluated_speaker']}")
        print(f"   Word Count: {results['student_word_count']}")
        print(f"   Total Speakers: {results['speaker_analysis']['total_speakers']}")

        # Content scores
        content = results["content_evaluation"]
        print(f"\n📋 Content Evaluation:")
        print(f"   Task Coverage: {content.get('task_coverage', 'N/A')}")
        print(f"   Grammar Control: {content.get('grammar_control', 'N/A')}")
        print(f"   Vocabulary Use: {content.get('vocabulary_use', 'N/A')}")
        print(f"   Logical Flow: {content.get('logical_flow', 'N/A')}")

        # Pronunciation details
        pronunciation = results["pronunciation"]
        print(f"\n🎤 Pronunciation Analysis:")
        print(f"   Average Confidence: {pronunciation['average_confidence']:.1%}")
        print(f"   Unclear Words: {pronunciation['unclear_count']}/{pronunciation['total_words']}")

        if pronunciation['unclear_words']:
            print(f"\n   Most Unclear Words:")
            for word in pronunciation['unclear_words'][:3]:
                print(f"   • '{word['word']}' ({word['confidence']:.1%})")

        # AI Insights
        insights = results["insights"]
        if insights['highlights']:
            print(f"\n🔍 Key Phrases Detected:")
            for highlight in insights['highlights']:
                print(f"   • {highlight.get('text', 'N/A')}")

        if insights['summary']:
            print(f"\n📄 AI Summary:")
            print(f"   {insights['summary']}")

        # Feedback
        if content.get('detailed_feedback'):
            print(f"\n💬 Detailed Feedback:")
            print(f"   {content['detailed_feedback']}")

        # Save results
        output_file = Path(audio_file).stem + "_assemblyai_evaluation.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n" + "="*70)
        print(f"💾 Complete results saved to: {output_file}")
        print("="*70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Tips:")
        print("   • Check your ASSEMBLYAI_API_KEY in .env file")
        print("   • Verify audio file format (mp3, wav, m4a)")
        print("   • Ensure stable internet connection")

if __name__ == "__main__":
    main()