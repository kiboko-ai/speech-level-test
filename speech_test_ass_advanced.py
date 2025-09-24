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

        # Advanced configuration (참조: Next.js API 코드)
        config = {
            "audio_url": upload_url,

            # Core features - 화자 구분 설정
            "speaker_labels": True,      # 화자 구분 활성화
            "speakers_expected": 2,       # 예상되는 화자 수 (선생님+학생)

            # Language and quality settings
            "language_detection": True,  # 자동 언어 감지
            "punctuate": True,           # 구두점 추가
            "format_text": True,         # 텍스트 포맷팅
            "disfluencies": True,        # "um", "ah" 등 캡처

            # Advanced analysis features
            "sentiment_analysis": True,  # 감정 분석
            "entity_detection": True,    # 개체명 인식
            "auto_highlights": True,     # 주요 구문 하이라이트

            # Quality boost
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

    def use_lemur_analysis(self, transcript_id: str, student_text: str, full_transcript: str,
                          course_level: int, student_speaker: str) -> Dict:
        """Use LeMUR (AI reasoning engine) for advanced analysis"""
        print("🤖 Analyzing with LeMUR AI reasoning engine...")

        # Advanced prompt matching the exact rubric format
        lemur_prompt = f"""You are an English speaking test evaluator.
Input: 1) Transcript of the current lesson, 2) Course level (1–4), 3) Student speaker role.

Full Transcript:
{full_transcript}

Student's Speech Only:
{student_text}

Student Speaker: {student_speaker}
Course Level: {course_level} (1-4 scale)

Task:
- Evaluate the student's performance in this single lesson using the rubric below.
- Score each sub-criterion from 1.0 to 5.0 in 0.1 increments.
- IMPORTANT: Adjust scoring based on course level {course_level}:
  • Level 1: Be very lenient. Basic attempts = good (3.5-4.0), Clear simple speech = excellent (4.5-5.0)
  • Level 2: Moderate. Simple accuracy = good (3.5-4.0), Some complexity = excellent (4.5-5.0)
  • Level 3: Higher standards. Varied accuracy = good (3.5-4.0), Natural fluency = excellent (4.5-5.0)
  • Level 4: Strict. Near-native = good (3.5-4.0), Professional = excellent (4.5-5.0)
- The same performance should receive DIFFERENT scores based on level.

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
4. Delivery
   - Pronunciation (clarity, understandable)
   - Intonation & Stress (natural rhythm, emphasis)

Also:
- Identify important vocabulary/phrases actually used (TOEIC, OPIc, academic/professional relevance).
- Provide concise feedback highlighting 1–2 strengths and 1–2 weaknesses.

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
  "pronunciation": <score>,
  "intonation_stress": <score>,
  "average_score": <average>,
  "vocab_phrases_used": [ "<word_or_phrase_1>", "<word_or_phrase_2>", ... ],
  "feedback": "<short summary of strengths and weaknesses>"
}}"""

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
        """Identify student speaker using AI-powered conversation context analysis"""

        if not utterances:
            return "A"

        # Get all speakers
        speakers = list(set(u['speaker'] for u in utterances))
        if len(speakers) == 1:
            return speakers[0]

        print("🔍 Using AI to analyze conversation context and identify roles...")

        # Prepare FULL conversation for AI analysis (not just first 20)
        conversation_text = ""
        for i, utterance in enumerate(utterances):
            speaker = utterance['speaker']
            text = utterance['text']
            conversation_text += f"{speaker}: {text}\n"
            # Include more context but cap at reasonable length
            if i >= 50 and len(conversation_text) > 5000:
                conversation_text += f"... (conversation continues for {len(utterances) - i} more utterances)\n"
                break

        # Use GPT-4 to analyze the conversation context
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        prompt = f"""You are analyzing an English language lesson conversation between a teacher and a student.
Your task is to identify which speaker (A or B) is the STUDENT based on the ENTIRE conversation context.

FULL CONVERSATION:
{conversation_text}

CRITICAL ANALYSIS POINTS:
1. Teaching patterns:
   - Who asks "How about...?", "What about...?", "Can you tell me...?" (teacher questions)
   - Who responds with attempts, answers, or "I think..." (student responses)
   - Who says "Good!", "That's right", "Very good", "Excellent" (teacher feedback)
   - Who receives corrections or guidance?

2. Language proficiency:
   - Who makes grammar mistakes or uses simpler language? (student)
   - Who speaks with perfect/native-like English? (teacher)
   - Who struggles with vocabulary or pronunciation? (student)
   - Who explains word meanings or grammar? (teacher)

3. Conversation flow:
   - Who leads the conversation and sets topics? (teacher)
   - Who follows instructions or prompts? (student)
   - Who asks for clarification about English? (student)
   - Who provides explanations? (teacher)

4. Common teacher phrases:
   - "Let's talk about...", "Now I want you to...", "Can you describe...?"
   - "That's correct", "Good job", "Try again"
   - Giving examples or model sentences

5. Common student patterns:
   - Hesitations: "um", "uh", "I think..."
   - Asking: "What does X mean?", "How do you say...?"
   - Making mistakes and being corrected
   - Shorter, simpler responses

IMPORTANT: Look at the OVERALL pattern across the ENTIRE conversation, not just individual utterances.
The student is the one LEARNING and PRACTICING English.
The teacher is the one TEACHING and GUIDING.

Output JSON:
{{
  "student_speaker": "A" or "B",
  "teacher_speaker": "A" or "B",
  "confidence": "high/medium/low",
  "student_evidence": ["specific quote or pattern", "another specific example"],
  "teacher_evidence": ["specific quote or pattern", "another specific example"],
  "reasoning": "Brief explanation of the overall pattern observed"
}}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing English language teaching conversations. You can accurately identify teachers vs students based on conversation patterns, language proficiency, and teaching dynamics."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,  # Low temperature for consistency
                max_tokens=500  # Increased for more detailed analysis
            )

            analysis = json.loads(response.choices[0].message.content)

            print("\n📊 AI Context Analysis:")
            print(f"   Student: Speaker {analysis['student_speaker']}")
            print(f"   Teacher: Speaker {analysis['teacher_speaker']}")
            print(f"   Confidence: {analysis['confidence']}")

            if analysis.get('student_evidence'):
                print(f"   Student Evidence:")
                for evidence in analysis['student_evidence'][:2]:
                    print(f"      - {evidence}")

            if analysis.get('teacher_evidence'):
                print(f"   Teacher Evidence:")
                for evidence in analysis['teacher_evidence'][:2]:
                    print(f"      - {evidence}")

            return analysis['student_speaker']

        except Exception as e:
            print(f"⚠️ AI analysis failed, using pattern-based fallback: {e}")

            # Fallback to pattern analysis if AI fails
            speaker_analysis = {}

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

    def generate_rubric_evaluation(self, student_text: str, full_transcript: str,
                                  course_level: int, student_speaker: str,
                                  pronunciation_score: float, intonation_score: float) -> Dict:
        """Generate evaluation using GPT-4 with the rubric format"""
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        prompt = f"""You are an English speaking test evaluator.

Full Transcript:
{full_transcript[:2000]}  # Limit for token management

Student's Speech Only:
{student_text[:1000]}

Student Speaker: {student_speaker}
Course Level: {course_level}

Pronunciation Score (already calculated): {pronunciation_score:.1f}
Intonation Score (already calculated): {intonation_score:.1f}

Task:
- Evaluate the student's performance using the rubric.
- Score each sub-criterion from 1.0 to 10.0 in 0.2 increments for detailed assessment.

IMPORTANT EVALUATION GUIDELINES:
1. For Content Relevance (task_coverage, appropriateness), Accuracy (grammar_control, vocabulary_use),
   and Coherence (logical_flow, cohesive_devices) criteria:
   - IGNORE hesitation markers like "uh", "um", "er", "ah", "hmm"
   - IGNORE filler words like isolated "a", "the" when used as hesitation
   - IGNORE repetitions caused by hesitation (e.g., "I, I, I went" → evaluate as "I went")
   - IGNORE false starts and self-corrections (e.g., "I want to- I would like to")
   - Focus ONLY on the actual content and message being conveyed
   - Evaluate grammar based on completed thoughts, not interrupted ones

2. EXCLUDE FROM EVALUATION (DO NOT PENALIZE):
   - Proper nouns (personal names, place names, brand names)
   - Pronunciation of names (e.g., McDonald's, Seoul, Jason, Maria)
   - Grammar of foreign names/places when used correctly in context
   - Cultural references that may be unfamiliar
   - Do NOT deduct points for mispronouncing or struggling with proper nouns

3. For Delivery (pronunciation, intonation_stress) criteria:
   - DO consider hesitations, repetitions, and fluency issues
   - These affect the delivery quality and should impact the score
   - BUT exclude proper noun pronunciation from scoring

- CRITICAL: Apply STRICT level-based scoring. The SAME performance MUST receive DIFFERENT scores:

For Level {course_level} students, use these SPECIFIC scoring guidelines:
""" + (
    """
Level 1 (Beginner - VERY LENIENT):
  • Any attempt to speak = 6.0-7.0
  • Basic understandable communication = 7.0-8.0
  • Clear simple sentences = 8.0-9.0
  • Perfect simple English = 9.0-10.0
  • Major grammar errors are acceptable if message is clear
  • Simple vocabulary is GOOD and should score HIGH
""" if course_level == 1 else
    """
Level 2 (Elementary - LENIENT):
  • Very basic attempts = 4.0-5.0
  • Basic communication with errors = 5.0-6.0
  • Simple but mostly accurate = 6.0-7.0
  • Some complexity attempted = 7.0-8.0
  • Good control of basics = 8.0-9.0
  • Some grammar errors acceptable
""" if course_level == 2 else
    """
Level 3 (Intermediate - MODERATE):
  • Too simple/basic = 3.0-4.0
  • Simple accuracy only = 4.0-5.0
  • Accurate with some variety = 5.0-6.0
  • Good range and control = 6.0-7.0
  • Natural fluency emerging = 7.0-8.0
  • Near-native in familiar topics = 8.0-10.0
""" if course_level == 3 else
    """
Level 4 (Advanced - VERY STRICT):
  • Basic level only = 2.0-3.0
  • Elementary/simple = 3.0-4.0
  • Intermediate level = 4.0-5.0
  • Some advanced features = 5.0-6.0
  • Good advanced control = 6.0-7.0
  • Near-native fluency = 7.0-8.0
  • Professional/native-like = 8.0-10.0
  • ANY grammar errors or awkwardness reduces score significantly
""") + f"""

Example: A student saying "I like McDonald's very much, I usually cook pork"
- Level 1: Would score 7.5-8.0 (clear basic communication - GOOD for beginner!)
- Level 2: Would score 5.5-6.0 (basic but understandable)
- Level 3: Would score 4.0-4.5 (too simple for intermediate)
- Level 4: Would score 2.5-3.0 (far below advanced expectations)

IMPORTANT for vocab_phrases_used:
- Extract INDIVIDUAL VOCABULARY WORDS (not phrases)
- Focus on: common nouns, verbs, adjectives, adverbs
- EXCLUDE:
  * Articles (a, the), pronouns (I, you, it), basic verbs (is, are, have), prepositions
  * PROPER NOUNS (names of people, places, brands, countries, cities)
  * Do NOT include: McDonald's, Seoul, Korea, Jason, etc.
- Prioritize: topic-specific words, descriptive words, academic vocabulary

Output JSON:
{{
  "task_coverage": <score>,
  "appropriateness": <score>,
  "grammar_control": <score>,
  "vocabulary_use": <score>,
  "logical_flow": <score>,
  "cohesive_devices": <score>,
  "vocab_phrases_used": ["temperature", "Celsius", "humid", "weather", "prefer"],
  "feedback": "1-2 strengths and 1-2 weaknesses"
}}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert English evaluator. Provide scores in 0.2 increments from 1.0-10.0 (e.g., 3.2, 3.4, 3.6) for detailed assessment on a 10-point scale."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"❌ GPT-4 evaluation error: {e}")
            # Return None to trigger error handling
            return None

    def compare_with_previous(self, current_eval: Dict, previous_eval: Dict, course_level: int) -> Dict:
        """Compare current evaluation with previous lesson to track progress"""

        from openai import OpenAI
        client = OpenAI()

        prompt = f"""You are an English speaking test evaluator comparing two lessons.

Input:
1) Current lesson evaluation: {json.dumps(current_eval)}
2) Previous lesson evaluation: {json.dumps(previous_eval)}
3) Course level: {course_level}

Task:
- Compare the current lesson with the previous lesson.
- Score the **change** (improvement or decline) in each sub-criterion.
- Use -2.0 to +2.0 scale where 0 = no change, positive = improvement, negative = decline.
- Course level sensitivity:
  • Levels 1–2: value clearer communication, more vocab, longer answers.
  • Levels 3–4: value better accuracy, complex structures, fluency, naturalness.

Also:
- Extract **new important VOCABULARY WORDS** (not full phrases) that appeared in this lesson but not in the previous lesson.
  Focus on: nouns, verbs, adjectives, adverbs - individual meaningful words
  Exclude: basic words like "is", "the", "a", common pronouns
  Prioritize: topic-specific vocabulary, academic words, descriptive words
- Highlight 1–2 key progress points and 1–2 areas still needing improvement.

Output JSON format:
{{
  "course_level": {course_level},
  "change_task_coverage": <change_score>,
  "change_appropriateness": <change_score>,
  "change_grammar_control": <change_score>,
  "change_vocabulary_use": <change_score>,
  "change_logical_flow": <change_score>,
  "change_cohesive_devices": <change_score>,
  "change_pronunciation": <change_score>,
  "change_intonation_stress": <change_score>,
  "average_change": <average>,
  "new_vocab_phrases": ["<single_word_1>", "<single_word_2>", ...],
  "progress_summary": "<short_text_summary>",
  "remaining_issues": "<short_text_summary>"
}}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert English evaluator tracking student progress."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"❌ Progress comparison error: {e}")
            return None

    def advanced_pronunciation_analysis(self, result: Dict, target_speaker: str) -> Dict:
        """Enhanced pronunciation analysis with Universal-1 data"""

        words = result.get('words', [])
        utterances = result.get('utterances', [])

        print(f"   📊 Debug - Total words: {len(words)}, Total utterances: {len(utterances)}")

        # Filter student words
        student_words = []
        student_utterances = [u for u in utterances if u['speaker'] == target_speaker]

        print(f"   📊 Student utterances found: {len(student_utterances)}")

        for utterance in student_utterances:
            start_time = utterance['start']
            end_time = utterance['end']

            # Find words in this utterance timeframe
            utterance_words = [
                w for w in words
                if start_time <= w.get('start', 0) <= end_time
            ]
            student_words.extend(utterance_words)

        print(f"   📊 Student words found: {len(student_words)}")

        if not student_words:
            print(f"   ⚠️ Warning: No student words with timing data found for speaker {target_speaker}")
            print(f"   ⚠️ Student utterances: {len(student_utterances)}, but no word-level timing data")
            return {
                "pronunciation_score": 3.0,
                "detailed_analysis": "No student words detected - possibly due to poor pronunciation or audio quality",
                "total_words": 0,
                "average_confidence": 0.5,
                "clarity_ratio": 0.5
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
            print(f"\n📊 Pronunciation Analysis:")
            pronunciation_analysis = self.advanced_pronunciation_analysis(result, student_speaker)

            # Step 7: LeMUR AI analysis (optional)
            lemur_analysis = {}
            if use_lemur and student_text.strip():
                full_transcript = result.get('text', '')
                lemur_analysis = self.use_lemur_analysis(
                    transcript_id,
                    student_text.strip(),
                    full_transcript,
                    course_level,
                    student_speaker
                )

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

            # Step 9: Create evaluation using rubric format with pronunciation metrics
            print(f"\n📊 Creating evaluation scores...")
            print(f"   Student text length: {len(student_text.split())} words")

            # Use pronunciation analysis for Delivery scores
            # Map 60%-100% to 1.0-5.0 (60% = 1.0, 80% = 3.0, 100% = 5.0)
            clarity = pronunciation_analysis.get('clarity_ratio', 0.5)
            confidence = pronunciation_analysis.get('average_confidence', 0.5)

            print(f"   📊 Clarity: {clarity:.1%}, Confidence: {confidence:.1%}")

            # No level-based adjustment for pronunciation scores
            # All levels use the same objective pronunciation scoring
            level_adjustment = 0

            # Base formula for 10-point scale: (metric - 0.5) * 20, clamped to 1.0-10.0
            # 50% → 1.0, 60% → 3.0, 70% → 5.0, 80% → 7.0, 90% → 9.0, 100% → 10.0
            base_pronunciation = (clarity - 0.5) * 20
            base_intonation = (confidence - 0.5) * 20

            # Apply level adjustment (no adjustment currently)
            pronunciation_score = max(1.0, min(10.0, base_pronunciation + level_adjustment))
            intonation_score = max(1.0, min(10.0, base_intonation + level_adjustment))

            # Generate GPT-4 evaluation if LeMUR not available
            print(f"\n📊 Generating GPT-4 evaluation...")

            if not lemur_analysis or 'task_coverage' not in lemur_analysis:
                print(f"   Using GPT-4 for evaluation (LeMUR not available or disabled)")
                gpt_evaluation = self.generate_rubric_evaluation(
                    student_text.strip(),
                    result.get('text', ''),
                    course_level,
                    student_speaker,
                    pronunciation_score,
                    intonation_score
                )
            else:
                print(f"   Using LeMUR evaluation results")
                gpt_evaluation = lemur_analysis

            # Ensure gpt_evaluation is not None
            if not gpt_evaluation:
                print(f"   ❌ GPT evaluation is None")
                print(f"   ❌ Student text preview: {student_text[:100]}...")
                print(f"   ❌ Full transcript length: {len(result.get('text', ''))}")
                raise Exception("GPT-4 evaluation failed: No evaluation data returned. This may be due to insufficient student speech content or API issues.")

            # Compile final results in rubric format
            return {
                # Metadata
                "audio_file": audio_path,
                "model_used": model_type if model_type else "default",
                "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),

                # Rubric scores (matching exact format)
                "course_level": course_level,
                "student_speaker": student_speaker,
                "task_coverage": gpt_evaluation.get('task_coverage', 3.0),
                "appropriateness": gpt_evaluation.get('appropriateness', 3.0),
                "grammar_control": gpt_evaluation.get('grammar_control', 3.0),
                "vocabulary_use": gpt_evaluation.get('vocabulary_use', 3.0),
                "logical_flow": gpt_evaluation.get('logical_flow', 3.0),
                "cohesive_devices": gpt_evaluation.get('cohesive_devices', 3.0),
                "pronunciation": round(pronunciation_score, 1),  # Round to 0.1 increments
                "intonation_stress": round(intonation_score, 1),  # Round to 0.1 increments
                "average_score": round(
                    (gpt_evaluation.get('task_coverage', 3.0) +
                     gpt_evaluation.get('appropriateness', 3.0) +
                     gpt_evaluation.get('grammar_control', 3.0) +
                     gpt_evaluation.get('vocabulary_use', 3.0) +
                     gpt_evaluation.get('logical_flow', 3.0) +
                     gpt_evaluation.get('cohesive_devices', 3.0) +
                     round(pronunciation_score, 1) +
                     round(intonation_score, 1)) / 8, 1),
                "vocab_phrases_used": gpt_evaluation.get('vocab_phrases_used', []),
                "feedback": gpt_evaluation.get('feedback', ''),

                # Additional detailed data
                "detailed_analysis": {
                    "student_word_count": len(student_text.split()),
                    "full_transcript": result.get('text', ''),
                    "student_text": student_text.strip(),
                    "speaker_distribution": speaker_words,
                    "pronunciation_details": pronunciation_analysis,
                    "assemblyai_insights": insights,
                    "model_performance": {
                        "overall_confidence": result.get('confidence', 0),
                        "language_detection_confidence": result.get('language_confidence', 0),
                        "processing_time": result.get('audio_duration', 0) / 1000
                    }
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

        # Display results in new rubric format
        print("\n" + "="*70)
        print("           ENGLISH SPEAKING EVALUATION RESULTS")
        print("="*70)

        print(f"\n📊 Course Level: {results['course_level']}")
        print(f"👤 Student: Speaker {results['student_speaker']}")
        print(f"🧠 Model: {results.get('model_used', 'Default')}")

        print("\n" + "-"*50)
        print("                    RUBRIC SCORES")
        print("-"*50)

        print("\n1. CONTENT RELEVANCE:")
        print(f"   • Task Coverage:      {results['task_coverage']:.1f}/10.0")
        print(f"   • Appropriateness:    {results['appropriateness']:.1f}/10.0")

        print("\n2. ACCURACY:")
        print(f"   • Grammar Control:    {results['grammar_control']:.1f}/10.0")
        print(f"   • Vocabulary Use:     {results['vocabulary_use']:.1f}/10.0")

        print("\n3. COHERENCE:")
        print(f"   • Logical Flow:       {results['logical_flow']:.1f}/10.0")
        print(f"   • Cohesive Devices:   {results['cohesive_devices']:.1f}/10.0")

        print("\n4. DELIVERY:")
        print(f"   • Pronunciation:      {results['pronunciation']:.1f}/10.0 (Clarity: {results['detailed_analysis']['pronunciation_details']['clarity_ratio']:.0%})")
        print(f"   • Intonation/Stress:  {results['intonation_stress']:.1f}/10.0 (Confidence: {results['detailed_analysis']['pronunciation_details']['average_confidence']:.0%})")

        print("\n" + "-"*50)
        print(f"📈 AVERAGE SCORE: {results['average_score']:.1f}/10.0")
        print("-"*50)

        # Vocabulary
        if results.get('vocab_phrases_used'):
            print(f"\n📝 IMPORTANT VOCABULARY/PHRASES USED:")
            for i, phrase in enumerate(results['vocab_phrases_used'][:10], 1):
                print(f"   {i}. {phrase}")

        # Feedback
        if results.get('feedback'):
            print(f"\n💬 FEEDBACK:")
            print(f"   {results['feedback']}")

        # Additional insights from AssemblyAI
        if results.get('detailed_analysis', {}).get('assemblyai_insights', {}).get('highlights'):
            print(f"\n🔍 KEY TOPICS DETECTED:")
            for highlight in results['detailed_analysis']['assemblyai_insights']['highlights']:
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