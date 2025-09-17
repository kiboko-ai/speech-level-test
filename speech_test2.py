import whisper
import torch
import numpy as np
from pyannote.audio import Pipeline
import json
import re
from typing import Dict, List, Tuple, Any
import librosa
import soundfile as sf
from pathlib import Path
import warnings
import os
from openai import OpenAI
from dataclasses import dataclass
from collections import defaultdict
import Levenshtein
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# Debug: Check if API key is loaded
loaded_key = os.getenv("OPENAI_API_KEY")
print(f"[DEBUG] Loaded API key from .env: {loaded_key[:20]}..." if loaded_key else "[DEBUG] No API key loaded from .env")
print(f"[DEBUG] .env file exists: {os.path.exists('.env')}")
print(f"[DEBUG] Current directory: {os.getcwd()}")

@dataclass
class PronunciationPattern:
    """Common pronunciation error patterns for Korean English learners"""
    confusions = {
        'r_l': [
            ('right', 'light'), ('rice', 'lice'), ('pray', 'play'),
            ('correct', 'collect'), ('problem', 'ploblem'),
            ('really', 'leally'), ('read', 'lead'), ('road', 'load'),
            ('arrive', 'alive'), ('crime', 'climb')
        ],
        'p_f': [
            ('coffee', 'copy'), ('office', 'oppice'), ('difficult', 'dippicult'),
            ('phone', 'pone'), ('photo', 'poto'), ('pharmacy', 'parmacy'),
            ('prefer', 'preper'), ('professional', 'propessional')
        ],
        'b_v': [
            ('very', 'berry'), ('video', 'bideo'), ('voice', 'boice'),
            ('vacation', 'bacation'), ('visit', 'bisit'), ('have', 'hab'),
            ('believe', 'belieb'), ('develop', 'debelop')
        ],
        'th': [
            ('think', 'sink'), ('thank', 'sank'), ('three', 'sree'),
            ('through', 'srough'), ('thought', 'sought'), ('thick', 'sick'),
            ('with', 'wis'), ('the', 'de'), ('this', 'dis'), ('that', 'dat')
        ],
        'z_s': [
            ('zoo', 'soo'), ('zero', 'sero'), ('zone', 'sone'),
            ('amazing', 'amasing'), ('organize', 'organise'),
            ('realize', 'realise'), ('analyze', 'analyse')
        ],
        'final_consonants': [
            ('want', 'wan'), ('just', 'jus'), ('first', 'firs'),
            ('last', 'las'), ('must', 'mus'), ('best', 'bes'),
            ('and', 'an'), ('friend', 'frien'), ('world', 'worl')
        ]
    }

class EnglishLevelEvaluator:
    def __init__(self, openai_api_key: str = None):
        """Initialize the evaluator with necessary models"""
        self.whisper_model = whisper.load_model("small")

        # Initialize OpenAI client
        final_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        print(f"[DEBUG] Initializing OpenAI with key: {final_key[:20]}..." if final_key else "[DEBUG] No API key found!")
        self.client = OpenAI(api_key=final_key)

        # Try to initialize speaker diarization (optional)
        # TEMPORARILY DISABLED for faster processing
        self.diarization_available = False
        print("Note: Using simple speaker diarization for faster processing")

        self.pronunciation_patterns = PronunciationPattern.confusions

    def transcribe_with_whisper(self, audio_path: str) -> Dict:
        """Transcribe audio using Whisper and get word-level timestamps"""
        result = self.whisper_model.transcribe(
            audio_path,
            language="en",
            word_timestamps=True,
            verbose=False
        )
        return result

    def perform_speaker_diarization(self, audio_path: str) -> Dict:
        """Perform speaker diarization to identify different speakers"""
        if not self.diarization_available:
            return self._simple_speaker_separation(audio_path)

        try:
            diarization = self.diarization_pipeline(audio_path)
            speakers = {}

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if speaker not in speakers:
                    speakers[speaker] = []
                speakers[speaker].append({
                    'start': turn.start,
                    'end': turn.end
                })

            return speakers
        except Exception as e:
            print(f"Diarization error: {e}")
            return self._simple_speaker_separation(audio_path)

    def _simple_speaker_separation(self, audio_path: str) -> Dict:
        """Simple speaker separation based on pause detection"""
        audio, sr = librosa.load(audio_path, sr=16000)

        # Detect speech segments using energy
        hop_length = 512
        frame_length = 2048
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

        # Threshold for speech detection
        threshold = np.percentile(energy, 30)
        speech_frames = energy > threshold

        # Group consecutive speech frames into segments
        segments = []
        in_speech = False
        start_time = 0

        for i, is_speech in enumerate(speech_frames):
            time = i * hop_length / sr

            if is_speech and not in_speech:
                start_time = time
                in_speech = True
            elif not is_speech and in_speech:
                if time - start_time > 0.5:  # Minimum segment duration
                    segments.append({'start': start_time, 'end': time})
                in_speech = False

        # Assign alternating speakers (simple heuristic)
        speakers = {'SPEAKER_0': [], 'SPEAKER_1': []}
        for i, segment in enumerate(segments):
            speaker = 'SPEAKER_0' if i % 2 == 0 else 'SPEAKER_1'
            speakers[speaker].append(segment)

        return speakers

    def map_transcript_to_speakers(self, transcript: Dict, speakers: Dict) -> List[Dict]:
        """Map transcript segments to identified speakers"""
        mapped_segments = []

        for segment in transcript.get('segments', []):
            segment_start = segment['start']
            segment_end = segment['end']

            # Find which speaker this segment belongs to
            assigned_speaker = None
            max_overlap = 0

            for speaker, intervals in speakers.items():
                for interval in intervals:
                    # Calculate overlap
                    overlap_start = max(segment_start, interval['start'])
                    overlap_end = min(segment_end, interval['end'])
                    overlap = max(0, overlap_end - overlap_start)

                    if overlap > max_overlap:
                        max_overlap = overlap
                        assigned_speaker = speaker

            mapped_segments.append({
                'speaker': assigned_speaker or 'UNKNOWN',
                'text': segment['text'],
                'start': segment_start,
                'end': segment_end,
                'words': segment.get('words', [])
            })

        return mapped_segments

    def analyze_pronunciation_errors(self, transcript_segments: List[Dict]) -> Dict:
        """Analyze potential pronunciation errors using AI and pattern matching"""
        # First, use manual pattern matching for known issues
        manual_issues = self._analyze_manual_patterns(transcript_segments)

        # Then, use AI for contextual pronunciation analysis
        ai_analysis = self._analyze_pronunciation_with_ai(transcript_segments)

        # Combine both analyses
        combined_issues = manual_issues['issues'].copy()
        if ai_analysis and 'issues' in ai_analysis:
            for error_type, issues in ai_analysis['issues'].items():
                if error_type not in combined_issues:
                    combined_issues[error_type] = []
                combined_issues[error_type].extend(issues)

        # Calculate final score (weighted average: 30% manual, 70% AI)
        manual_score = manual_issues['score']
        ai_score = ai_analysis.get('score', manual_score) if ai_analysis else manual_score
        final_score = (manual_score * 0.3 + ai_score * 0.7)

        return {
            'score': round(final_score * 2) / 2,  # Round to 0.5 increments
            'issues': combined_issues,
            'error_rate': manual_issues['error_rate'],
            'total_words': manual_issues['total_words'],
            'ai_feedback': ai_analysis.get('feedback', '') if ai_analysis else ''
        }

    def _analyze_manual_patterns(self, transcript_segments: List[Dict]) -> Dict:
        """Analyze pronunciation using manual patterns"""
        pronunciation_issues = defaultdict(list)
        total_words = 0

        for segment in transcript_segments:
            words = segment.get('words', [])
            if not words:
                words = [{'word': w} for w in segment['text'].split()]

            for word_info in words:
                word = word_info.get('word', '').lower().strip('.,!?;:')
                if not word:
                    continue

                total_words += 1

                for error_type, patterns in self.pronunciation_patterns.items():
                    for correct, incorrect in patterns:
                        if self._is_likely_mispronunciation(word, incorrect, correct):
                            pronunciation_issues[error_type].append({
                                'transcribed': word,
                                'likely_intended': correct,
                                'error_type': error_type,
                                'segment': segment['text'][:100]
                            })

        error_count = sum(len(issues) for issues in pronunciation_issues.values())
        error_rate = error_count / max(total_words, 1)
        pronunciation_score = max(1.0, min(5.0, 5.0 - (error_rate * 20)))

        return {
            'score': round(pronunciation_score * 2) / 2,
            'issues': dict(pronunciation_issues),
            'error_rate': error_rate,
            'total_words': total_words
        }

    def _analyze_pronunciation_with_ai(self, transcript_segments: List[Dict]) -> Dict:
        """Use AI to analyze pronunciation errors contextually"""
        student_text = " ".join([seg['text'] for seg in transcript_segments])

        if not student_text.strip():
            return None

        prompt = f"""You are an expert in analyzing pronunciation errors for Korean English learners.

Analyze this transcribed speech for pronunciation errors. The text below is from automatic speech recognition (ASR) of a Korean student speaking English. ASR often mistranscribes words when pronunciation is unclear.

Student's transcribed speech:
{student_text}

Task:
1. Identify words that seem to be mistranscribed due to pronunciation errors
2. Consider typical Korean learner pronunciation patterns:
   - r/l confusion (rice→lice, problem→ploblem)
   - p/f confusion (coffee→copy, office→oppice)
   - b/v confusion (very→berry, believe→belieb)
   - th sounds (think→sink, with→wis)
   - z/s confusion (zero→sero)
   - Final consonant deletion (want→wan, friend→frien)
   - Vowel substitutions

3. Look for:
   - Words that don't make sense in context (likely mistranscriptions)
   - Unusual spellings or non-words
   - Grammatically correct but semantically odd word choices
   - Common Korean pronunciation error patterns

4. Score pronunciation from 1.0 to 5.0:
   - 5.0: Native-like, clear pronunciation
   - 4.0-4.5: Minor errors, easily understood
   - 3.0-3.5: Noticeable errors but comprehensible
   - 2.0-2.5: Significant errors affecting understanding
   - 1.0-1.5: Severe errors, difficult to understand

Output JSON:
{{
  "score": <float>,
  "issues": {{
    "contextual_errors": [
      {{
        "transcribed": "<word in transcript>",
        "likely_intended": "<probable intended word>",
        "reason": "<why this is likely an error>",
        "context": "<surrounding text>"
      }}
    ],
    "pronunciation_patterns": [
      {{
        "pattern": "<type of error>",
        "examples": ["<example1>", "<example2>"],
        "impact": "<low/medium/high>"
      }}
    ]
  }},
  "feedback": "<Brief explanation of main pronunciation issues>"
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert in English pronunciation assessment for Korean learners."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )

            result = json.loads(response.choices[0].message.content)

            # Transform AI response to match our format
            transformed_issues = {}

            if 'issues' in result:
                if 'contextual_errors' in result['issues']:
                    transformed_issues['ai_contextual'] = result['issues']['contextual_errors']

                if 'pronunciation_patterns' in result['issues']:
                    for pattern in result['issues']['pronunciation_patterns']:
                        pattern_type = pattern.get('pattern', 'unknown').lower().replace(' ', '_')
                        if pattern_type not in transformed_issues:
                            transformed_issues[pattern_type] = []
                        for example in pattern.get('examples', []):
                            transformed_issues[pattern_type].append({
                                'transcribed': example,
                                'error_type': pattern_type,
                                'impact': pattern.get('impact', 'medium')
                            })

            return {
                'score': result.get('score', 3.0),
                'issues': transformed_issues,
                'feedback': result.get('feedback', '')
            }

        except Exception as e:
            print(f"AI pronunciation analysis error: {e}")
            return None

    def _is_likely_mispronunciation(self, transcribed: str, error_pattern: str, correct_pattern: str) -> bool:
        """Check if a transcribed word is likely a mispronunciation"""
        # Use Levenshtein distance to check similarity
        if Levenshtein.distance(transcribed, error_pattern) <= 1:
            return True

        # Check if the transcribed word contains the error pattern
        if error_pattern in transcribed and correct_pattern not in transcribed:
            return True

        return False

    def evaluate_with_gpt(self, transcript: str, speaker_segments: List[Dict],
                         course_level: int = 2, student_speaker: str = "SPEAKER_1") -> Dict:
        """Use GPT-4 to evaluate the transcript based on the rubric"""

        # Extract only student's speech
        student_text = " ".join([
            seg['text'] for seg in speaker_segments
            if seg['speaker'] == student_speaker
        ])

        # Prepare the evaluation prompt
        prompt = f"""You are an English speaking test evaluator.
Input:
1) Student's transcript: {student_text}
2) Course level: {course_level}
3) Full conversation context: {transcript}

Task:
- Evaluate the student's performance using the rubric below.
- Score each sub-criterion from 1.0 to 5.0 in 0.5 increments.
- Take into account the student's course level when judging:
  " Levels 12: focus on communication clarity, basic vocabulary growth, task completion.
  " Levels 34: focus on accuracy, complex structures, fluency, and natural expression.

Rubric categories:
1. Content Relevance
   - Task Coverage (answering the prompt fully)
   - Appropriateness (staying on-topic)
2. Accuracy
   - Grammar Control (sentence correctness, tense, structure)
   - Vocabulary Use (range, appropriateness, precision)
3. Coherence
   - Logical Flow (clear reasoning, structured ideas)
   - Cohesive Devices (use of linking words, transitions)

Also:
- Identify important vocabulary/phrases actually used (TOEIC, OPIc, academic/professional relevance).
- Provide concise feedback highlighting 12 strengths and 12 weaknesses.

Output JSON format:
{{
  "course_level": {course_level},
  "student_speaker": "{student_speaker}",
  "task_coverage": <score>,
  "appropriateness": <score>,
  "grammar_control": <score>,
  "vocabulary_use": <score>,
  "logical_flow": <score>,
  "cohesive_devices": <score>,
  "vocab_phrases_used": ["<word_or_phrase_1>", "<word_or_phrase_2>", ...],
  "feedback": "<short summary of strengths and weaknesses>"
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert English language evaluator."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )

            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"GPT evaluation error: {e}")
            return self._generate_default_scores(course_level, student_speaker)

    def _generate_default_scores(self, course_level: int, student_speaker: str) -> Dict:
        """Generate default scores if GPT evaluation fails"""
        return {
            "course_level": course_level,
            "student_speaker": student_speaker,
            "task_coverage": 3.0,
            "appropriateness": 3.0,
            "grammar_control": 3.0,
            "vocabulary_use": 3.0,
            "logical_flow": 3.0,
            "cohesive_devices": 3.0,
            "vocab_phrases_used": [],
            "feedback": "Evaluation could not be completed automatically."
        }

    def identify_student_by_context(self, speaker_segments: List[Dict]) -> str:
        """Identify student speaker using conversation context and patterns"""

        if not speaker_segments:
            return "SPEAKER_00"

        # Group segments by speaker
        speakers_text = defaultdict(list)
        speakers_word_count = defaultdict(int)

        for segment in speaker_segments:
            speaker = segment.get('speaker', 'UNKNOWN')
            text = segment.get('text', '')
            speakers_text[speaker].append(text)
            speakers_word_count[speaker] += len(text.split())

        if len(speakers_text) == 1:
            return list(speakers_text.keys())[0]

        print("\n🔍 Analyzing conversation context to identify student...")

        speaker_analysis = {}

        for speaker in speakers_text:
            all_text = " ".join(speakers_text[speaker])

            # Teacher indicators (질문, 지시, 피드백 패턴)
            teacher_patterns = [
                # Questions and prompts
                r'\b(what|where|when|why|how|can you|could you|would you|do you|did you|will you|are you|is it)\b',
                r'\b(tell me|explain|describe|talk about|speak about|share|discuss)\b',
                # Instructions and transitions
                r'\b(let\'s|now|next|first|then|finally|okay|alright|so)\b',
                # Feedback and evaluation
                r'\b(good|great|excellent|nice|well done|correct|right|yes|that\'s right|exactly|perfect)\b',
                r'\b(actually|no|not quite|try again|listen|repeat|be careful)\b',
                # Teaching language
                r'\b(lesson|homework|assignment|practice|study|learn|understand|remember|focus on)\b',
                r'\b(example|for instance|such as|like|means|definition)\b',
                # Control phrases
                r'\b(please|thank you|you should|you need to|you have to|you must)\b'
            ]

            # Student indicators (답변, 불확실성, 학습자 언어)
            student_patterns = [
                # Uncertainty and hesitation
                r'\b(I think|maybe|perhaps|probably|I guess|I\'m not sure|I don\'t know)\b',
                r'\b(um|uh|er|ah|hmm|well|you know|I mean|sort of|kind of)\b',
                # Learning language
                r'\b(I learned|I studied|I practiced|I tried|I want to|I need to|I have to)\b',
                r'\b(difficult|hard|easy|confusing|confused|understand|don\'t understand)\b',
                # Personal experiences and opinions
                r'\b(I like|I don\'t like|I prefer|I enjoy|I hate|I love)\b',
                r'\b(my|me|myself|mine|I am|I\'m|I was|I have|I\'ve)\b',
                # Response patterns
                r'^(yes|yeah|no|okay|sure|of course|right)\b',
                # Student-specific content
                r'\b(weekend|vacation|friend|family|home|school|class|test)\b'
            ]

            import re

            # Count pattern matches
            teacher_score = 0
            student_score = 0

            # Weight each pattern category differently
            for i, pattern in enumerate(teacher_patterns):
                matches = len(re.findall(pattern, all_text.lower()))
                # Give more weight to question patterns
                weight = 2 if i < 2 else 1
                teacher_score += matches * weight

            for i, pattern in enumerate(student_patterns):
                matches = len(re.findall(pattern, all_text.lower()))
                # Give more weight to uncertainty markers
                weight = 2 if i < 2 else 1
                student_score += matches * weight

            # Analyze conversation dynamics
            # First speaker bonus for teachers (they usually initiate)
            if speaker_segments and speaker == speaker_segments[0].get('speaker'):
                teacher_score += 10

            # Count questions
            questions = len(re.findall(r'\?', all_text))
            teacher_score += questions * 3

            # Count exclamations (teachers use more for encouragement)
            exclamations = len(re.findall(r'!', all_text))
            teacher_score += exclamations

            # Average utterance length
            avg_words = speakers_word_count[speaker] / max(1, len(speakers_text[speaker]))
            if avg_words > 20:  # Longer explanations suggest teacher
                teacher_score += 5
            elif avg_words < 10:  # Shorter responses suggest student
                student_score += 5

            # Sentence complexity (teachers use more complex sentences)
            complex_connectors = len(re.findall(
                r'\b(however|therefore|furthermore|moreover|although|because|since|while)\b',
                all_text.lower()
            ))
            teacher_score += complex_connectors * 2

            speaker_analysis[speaker] = {
                'teacher_score': teacher_score,
                'student_score': student_score,
                'word_count': speakers_word_count[speaker],
                'avg_utterance_length': avg_words,
                'questions': questions,
                'likely_role': 'teacher' if teacher_score > student_score else 'student',
                'confidence': abs(teacher_score - student_score),
                'sample': speakers_text[speaker][0][:100] if speakers_text[speaker] else ""
            }

        # Display analysis results
        print("\n📊 Context-Based Speaker Analysis:")
        print("-" * 50)
        for speaker, data in speaker_analysis.items():
            role = data['likely_role']
            t_score = data['teacher_score']
            s_score = data['student_score']
            words = data['word_count']
            conf = data['confidence']
            print(f"   {speaker}:")
            print(f"      Role: {role.title()} (confidence: {conf})")
            print(f"      Scores: Teacher={t_score}, Student={s_score}")
            print(f"      Words: {words}, Questions: {data['questions']}")
            print(f"      Sample: \"{data['sample'][:60]}...\"")
        print("-" * 50)

        # Find student with highest confidence
        student_candidates = [(speaker, data) for speaker, data in speaker_analysis.items()
                            if data['likely_role'] == 'student']

        if student_candidates:
            # Sort by confidence and return the most likely student
            student_candidates.sort(key=lambda x: x[1]['confidence'], reverse=True)
            return student_candidates[0][0]
        else:
            # Fallback: choose speaker with lowest teacher score
            return min(speaker_analysis.keys(),
                      key=lambda s: speaker_analysis[s]['teacher_score'])

    def analyze_prosody(self, audio_path: str, speaker_segments: List[Dict],
                       student_speaker: str = "SPEAKER_1") -> Dict:
        """Analyze prosody (intonation and stress) using librosa"""
        audio, sr = librosa.load(audio_path, sr=16000)

        # Extract prosodic features for student segments
        student_prosody_features = []

        for segment in speaker_segments:
            if segment['speaker'] != student_speaker:
                continue

            start_sample = int(segment['start'] * sr)
            end_sample = int(segment['end'] * sr)
            segment_audio = audio[start_sample:end_sample]

            if len(segment_audio) < sr * 0.1:  # Skip very short segments
                continue

            # Extract pitch (F0)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                segment_audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7')
            )

            # Calculate pitch statistics
            valid_f0 = f0[~np.isnan(f0)]
            if len(valid_f0) > 0:
                pitch_range = np.ptp(valid_f0)
                pitch_std = np.std(valid_f0)

                student_prosody_features.append({
                    'pitch_range': pitch_range,
                    'pitch_std': pitch_std,
                    'duration': segment['end'] - segment['start']
                })

        if not student_prosody_features:
            return {'score': 3.0, 'analysis': 'Insufficient audio data for prosody analysis'}

        # Calculate prosody score based on variation (more variation = better intonation)
        avg_pitch_range = np.mean([f['pitch_range'] for f in student_prosody_features])
        avg_pitch_std = np.mean([f['pitch_std'] for f in student_prosody_features])

        # Normalize to score (1.0 to 5.0)
        # Good intonation has pitch range of 50-200 Hz and std of 20-60 Hz
        range_score = min(5.0, max(1.0, (avg_pitch_range / 40)))
        std_score = min(5.0, max(1.0, (avg_pitch_std / 12)))

        prosody_score = (range_score + std_score) / 2
        prosody_score = round(prosody_score * 2) / 2  # Round to 0.5 increments

        return {
            'score': prosody_score,
            'analysis': {
                'avg_pitch_range': float(avg_pitch_range),
                'avg_pitch_std': float(avg_pitch_std),
                'segments_analyzed': len(student_prosody_features)
            }
        }

    def evaluate_audio(self, audio_path: str, course_level: int = 2) -> Dict:
        """Main evaluation function"""
        print("Step 1: Transcribing audio with Whisper...")
        transcript_result = self.transcribe_with_whisper(audio_path)
        full_transcript = transcript_result['text']

        print("Step 2: Performing speaker diarization...")
        speakers = self.perform_speaker_diarization(audio_path)

        print("Step 3: Mapping transcript to speakers...")
        speaker_segments = self.map_transcript_to_speakers(transcript_result, speakers)

        # Identify student speaker using context analysis
        student_speaker = self.identify_student_by_context(speaker_segments)

        print(f"Step 4: Identified student as {student_speaker}")

        print("Step 5: Evaluating content, accuracy, and coherence with GPT...")
        gpt_evaluation = self.evaluate_with_gpt(
            full_transcript, speaker_segments, course_level, student_speaker
        )

        print("Step 6: Analyzing pronunciation errors...")
        pronunciation_analysis = self.analyze_pronunciation_errors(
            [seg for seg in speaker_segments if seg['speaker'] == student_speaker]
        )

        print("Step 7: Analyzing prosody (intonation & stress)...")
        prosody_analysis = self.analyze_prosody(audio_path, speaker_segments, student_speaker)

        # Combine all scores
        final_evaluation = gpt_evaluation.copy()
        final_evaluation['pronunciation'] = pronunciation_analysis['score']
        final_evaluation['intonation_stress'] = prosody_analysis['score']

        # Calculate average score
        score_fields = [
            'task_coverage', 'appropriateness', 'grammar_control',
            'vocabulary_use', 'logical_flow', 'cohesive_devices',
            'pronunciation', 'intonation_stress'
        ]

        scores = [final_evaluation.get(field, 3.0) for field in score_fields]
        final_evaluation['average_score'] = round(sum(scores) / len(scores), 1)

        # Add detailed analysis
        final_evaluation['detailed_analysis'] = {
            'pronunciation_issues': pronunciation_analysis['issues'],
            'pronunciation_error_rate': pronunciation_analysis['error_rate'],
            'pronunciation_ai_feedback': pronunciation_analysis.get('ai_feedback', ''),
            'prosody_metrics': prosody_analysis['analysis'],
            'speaker_distribution': dict(speaker_word_counts),
            'total_duration': transcript_result.get('duration', 0)
        }

        return final_evaluation

def main():
    # Initialize evaluator
    evaluator = EnglishLevelEvaluator()

    # Path to audio file
    audio_file = '/Users/jason/Downloads/35526.m4a'  # Using original m4a

    # Check if file exists
    if not Path(audio_file).exists():
        print(f"Error: Audio file not found at {audio_file}")
        return

    print(f"Evaluating audio file: {audio_file}")
    print("-" * 50)

    # Perform evaluation
    try:
        results = evaluator.evaluate_audio(audio_file, course_level=2)

        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)

        print(f"\nStudent Speaker: {results['student_speaker']}")
        print(f"Course Level: {results['course_level']}")
        print("\nSCORES (1.0-5.0):")
        print("-" * 30)
        print(f"Content Relevance:")
        print(f"  - Task Coverage: {results.get('task_coverage', 'N/A')}")
        print(f"  - Appropriateness: {results.get('appropriateness', 'N/A')}")
        print(f"\nAccuracy:")
        print(f"  - Grammar Control: {results.get('grammar_control', 'N/A')}")
        print(f"  - Vocabulary Use: {results.get('vocabulary_use', 'N/A')}")
        print(f"\nCoherence:")
        print(f"  - Logical Flow: {results.get('logical_flow', 'N/A')}")
        print(f"  - Cohesive Devices: {results.get('cohesive_devices', 'N/A')}")
        print(f"\nDelivery:")
        print(f"  - Pronunciation: {results.get('pronunciation', 'N/A')}")
        print(f"  - Intonation & Stress: {results.get('intonation_stress', 'N/A')}")
        print("-" * 30)
        print(f"\nAVERAGE SCORE: {results['average_score']}")

        print(f"\nVocabulary/Phrases Used:")
        for vocab in results.get('vocab_phrases_used', [])[:10]:
            print(f"  - {vocab}")

        print(f"\nFeedback:")
        print(f"{results.get('feedback', 'No feedback available')}")

        if 'detailed_analysis' in results:
            print("\n" + "="*50)
            print("DETAILED ANALYSIS")
            print("="*50)

            if results['detailed_analysis']['pronunciation_issues']:
                print("\nPronunciation Issues Detected:")
                for error_type, issues in results['detailed_analysis']['pronunciation_issues'].items():
                    if error_type == 'ai_contextual':
                        print(f"\n  AI-Detected Contextual Errors:")
                        for issue in issues[:3]:
                            print(f"    - '{issue.get('transcribed', '')}' → '{issue.get('likely_intended', '')}'")
                            if 'reason' in issue:
                                print(f"      Reason: {issue['reason']}")
                    else:
                        print(f"\n  {error_type.replace('_', '/')} pattern:")
                        for issue in issues[:3]:
                            if 'likely_intended' in issue:
                                print(f"    - '{issue['transcribed']}' → '{issue['likely_intended']}'")
                            else:
                                print(f"    - '{issue.get('transcribed', '')}'")

            if results['detailed_analysis'].get('pronunciation_ai_feedback'):
                print(f"\nAI Pronunciation Analysis:")
                print(f"  {results['detailed_analysis']['pronunciation_ai_feedback']}")

            print(f"\nPronunciation Error Rate: {results['detailed_analysis']['pronunciation_error_rate']:.2%}")
            print(f"\nSpeaker Word Distribution:")
            for speaker, count in results['detailed_analysis']['speaker_distribution'].items():
                print(f"  - {speaker}: {count} words")

        # Save results to JSON
        output_file = audio_file.replace('.m4a', '_evaluation.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nFull results saved to: {output_file}")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()