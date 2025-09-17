import re
from pathlib import Path
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import Dict, List
from collections import defaultdict
import Levenshtein

load_dotenv()

# Debug: Check if API key is loaded
loaded_key = os.getenv("OPENAI_API_KEY")
print(f"[DEBUG] Loaded API key: {loaded_key[:20]}..." if loaded_key else "[DEBUG] No API key loaded")

class SRTEvaluator:
    def __init__(self):
        """Initialize with OpenAI client"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        self.client = OpenAI(api_key=api_key)

        # Korean learner pronunciation patterns
        self.pronunciation_patterns = {
            'r_l': [
                ('right', 'light'), ('rice', 'lice'), ('pray', 'play'),
                ('correct', 'collect'), ('problem', 'ploblem'),
                ('really', 'leally'), ('read', 'lead'), ('road', 'load')
            ],
            'p_f': [
                ('coffee', 'copy'), ('office', 'oppice'), ('difficult', 'dippicult'),
                ('phone', 'pone'), ('photo', 'poto')
            ],
            'b_v': [
                ('very', 'berry'), ('video', 'bideo'), ('voice', 'boice'),
                ('vacation', 'bacation'), ('have', 'hab')
            ],
            'th': [
                ('think', 'sink'), ('thank', 'sank'), ('three', 'sree'),
                ('with', 'wis'), ('the', 'de'), ('this', 'dis')
            ],
            'final_consonants': [
                ('want', 'wan'), ('just', 'jus'), ('first', 'firs'),
                ('and', 'an'), ('friend', 'frien'), ('world', 'worl')
            ]
        }

    def parse_srt(self, srt_path: str) -> List[Dict]:
        """Parse SRT file with speaker information"""
        segments = []

        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split by double newline to get subtitle blocks
        blocks = content.strip().split('\n\n')

        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                # Parse timestamp
                timestamp_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', lines[1])
                if timestamp_match:
                    start_time = timestamp_match.group(1)
                    end_time = timestamp_match.group(2)

                    # Parse text with speaker
                    text = ' '.join(lines[2:])

                    # Extract speaker (formats: [Speaker], Speaker:, <Speaker>)
                    speaker_match = (
                        re.match(r'\[(.*?)\]\s*(.*)', text) or
                        re.match(r'<(.*?)>\s*(.*)', text) or
                        re.match(r'(.*?):\s*(.*)', text)
                    )

                    if speaker_match:
                        speaker = speaker_match.group(1).strip()
                        clean_text = speaker_match.group(2).strip()
                    else:
                        speaker = "UNKNOWN"
                        clean_text = text

                    segments.append({
                        'speaker': speaker,
                        'text': clean_text,
                        'start': start_time,
                        'end': end_time
                    })

        return segments

    def identify_student(self, segments: List[Dict]) -> str:
        """Identify student speaker from segments"""
        speaker_counts = {}
        for segment in segments:
            speaker = segment['speaker']
            if speaker not in speaker_counts:
                speaker_counts[speaker] = 0
            speaker_counts[speaker] += len(segment['text'].split())

        # Common identifiers
        teacher_keywords = ['teacher', 'instructor', 'tutor', 'professor', 't', 'T']
        student_keywords = ['student', 'learner', 's', 'S', 'pupil']

        for speaker in speaker_counts:
            speaker_lower = speaker.lower()
            if any(keyword in speaker_lower for keyword in student_keywords):
                return speaker
            elif any(keyword in speaker_lower for keyword in teacher_keywords):
                continue

        # If no clear identifier, assume student speaks less
        if speaker_counts:
            return min(speaker_counts.keys(), key=lambda k: speaker_counts[k])
        return "UNKNOWN"

    def analyze_pronunciation_patterns(self, student_segments: List[Dict]) -> Dict:
        """Analyze pronunciation errors from transcript"""
        issues = defaultdict(list)
        total_words = 0

        for segment in student_segments:
            words = segment['text'].lower().split()
            total_words += len(words)

            for word in words:
                word = word.strip('.,!?;:')
                # Check against known patterns
                for error_type, patterns in self.pronunciation_patterns.items():
                    for correct, incorrect in patterns:
                        if Levenshtein.distance(word, incorrect) <= 1:
                            issues[error_type].append({
                                'transcribed': word,
                                'likely_intended': correct,
                                'segment': segment['text'][:50]
                            })

        error_count = sum(len(v) for v in issues.values())
        error_rate = error_count / max(total_words, 1)

        return {
            'issues': dict(issues),
            'error_rate': error_rate,
            'total_words': total_words
        }

    def evaluate_with_gpt(self, segments: List[Dict], student_speaker: str,
                         course_level: int = 2) -> Dict:
        """Comprehensive evaluation using GPT-4"""

        # Extract student segments
        student_segments = [seg for seg in segments if seg['speaker'] == student_speaker]
        student_text = " ".join([seg['text'] for seg in student_segments])

        # Create dialogue context
        full_dialogue = "\n".join([f"{seg['speaker']}: {seg['text']}" for seg in segments])

        # Analyze pronunciation patterns locally
        pronunciation_analysis = self.analyze_pronunciation_patterns(student_segments)

        prompt = f"""You are an expert English speaking test evaluator. Evaluate the student's performance comprehensively.

Student's speech (extracted from dialogue):
{student_text}

Full dialogue context:
{full_dialogue}

Course level: {course_level}
- Levels 1-2: focus on communication clarity, basic vocabulary growth, task completion
- Levels 3-4: focus on accuracy, complex structures, fluency, and natural expression

Pronunciation issues detected (for reference):
{json.dumps(pronunciation_analysis['issues'], indent=2)}
Error rate: {pronunciation_analysis['error_rate']:.2%}

Task: Score each criterion from 1.0 to 5.0 in 0.1 increments (e.g., 3.2, 3.7, 4.1).

Rubric categories:
1. Content Relevance
   - task_coverage: How well the student addresses the topic/questions
   - appropriateness: Staying on-topic and appropriate responses

2. Accuracy
   - grammar_control: Sentence correctness, tense usage, structure
   - vocabulary_use: Range, appropriateness, precision of words

3. Coherence
   - logical_flow: Clear reasoning, structured ideas
   - cohesive_devices: Use of linking words, transitions

4. Delivery (estimate from transcript)
   - pronunciation: Based on misspellings/errors suggesting pronunciation issues
   - intonation_stress: Estimate based on sentence structure and punctuation

Also identify:
- Key vocabulary/phrases used (especially TOEIC/OPIc relevant)
- Main strengths and weaknesses
- Specific improvement suggestions

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
  "average_score": <calculated average>,
  "vocab_phrases_used": ["phrase1", "phrase2", ...],
  "feedback": "Detailed feedback with strengths and weaknesses",
  "improvement_suggestions": ["suggestion1", "suggestion2", ...]
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

            result = json.loads(response.choices[0].message.content)

            # Add pronunciation analysis details
            result['pronunciation_analysis'] = {
                'error_rate': pronunciation_analysis['error_rate'],
                'total_words': pronunciation_analysis['total_words'],
                'issues_found': pronunciation_analysis['issues']
            }

            return result

        except Exception as e:
            print(f"GPT evaluation error: {e}")
            return self._generate_default_scores(course_level, student_speaker)

    def _generate_default_scores(self, course_level: int, student_speaker: str) -> Dict:
        """Generate default scores if GPT fails"""
        return {
            "course_level": course_level,
            "student_speaker": student_speaker,
            "task_coverage": 3.0,
            "appropriateness": 3.0,
            "grammar_control": 3.0,
            "vocabulary_use": 3.0,
            "logical_flow": 3.0,
            "cohesive_devices": 3.0,
            "pronunciation": 3.0,
            "intonation_stress": 3.0,
            "average_score": 3.0,
            "vocab_phrases_used": [],
            "feedback": "Evaluation could not be completed automatically.",
            "improvement_suggestions": []
        }

def main():
    print("="*50)
    print("SRT-based English Level Evaluator")
    print("="*50)

    # Default values for quick testing
    default_srt = '/Users/jason/Downloads/35526.srt'
    default_student = 'B'
    default_level = 2

    # Get SRT file path
    srt_input = input(f"\nEnter SRT file path (default: {default_srt}): ").strip().strip("'\"")
    srt_file = srt_input if srt_input else default_srt

    if not Path(srt_file).exists():
        print(f"Error: File not found at {srt_file}")
        return

    # Get course level
    try:
        level_input = input(f"Enter course level 1-4 (default: {default_level}): ").strip()
        course_level = int(level_input) if level_input else default_level
        if course_level not in [1, 2, 3, 4]:
            course_level = default_level
    except:
        course_level = default_level

    # Get student speaker (A or B)
    student_input = input(f"Enter student speaker A or B (default: {default_student}): ").strip().upper()
    if student_input in ['A', 'B']:
        student_choice = student_input
    else:
        student_choice = default_student

    # Ask if user wants detailed pronunciation analysis
    detail_input = input("Show detailed pronunciation errors? (y/N): ").strip().lower()
    show_detailed = detail_input == 'y'

    # Map to actual speaker names
    speaker_mapping = {
        'A': 'SpeakerA',
        'B': 'SpeakerB'
    }

    evaluator = SRTEvaluator()

    print(f"\n📂 Processing: {Path(srt_file).name}")
    print("-" * 50)

    # Parse SRT
    print("Step 1: Parsing SRT file...")
    segments = evaluator.parse_srt(srt_file)
    print(f"  ✓ Found {len(segments)} segments")

    # Identify speakers
    speakers = list(set(seg['speaker'] for seg in segments))
    print(f"  ✓ Speakers: {', '.join(speakers)}")

    # Identify student
    print("\nStep 2: Identifying student speaker...")

    # Use the selected student speaker
    student_speaker = speaker_mapping.get(student_choice, f'Speaker{student_choice}')

    # Verify the speaker exists in the SRT
    if student_speaker in speakers:
        print(f"  ✓ Student speaker: {student_speaker} (User selected: {student_choice})")
    else:
        # Try alternative formats
        alt_formats = [f'Speaker{student_choice}', f'speaker{student_choice}',
                       f'SPEAKER{student_choice}', student_choice]
        found = False
        for alt in alt_formats:
            if alt in speakers:
                student_speaker = alt
                print(f"  ✓ Student speaker found as: {student_speaker}")
                found = True
                break

        if not found:
            print(f"  ⚠️ Warning: {student_speaker} not found in SRT")
            print(f"  Available speakers: {', '.join(speakers)}")
            # Fallback to auto-detection
            student_speaker = evaluator.identify_student(segments)
            print(f"  ✓ Auto-detected student as: {student_speaker}")

    # Count words
    student_segments = [seg for seg in segments if seg['speaker'] == student_speaker]
    student_words = sum(len(seg['text'].split()) for seg in student_segments)
    print(f"  ✓ Student speech: {student_words} words")

    # Evaluate
    print("\nStep 3: Evaluating with GPT-4...")
    results = evaluator.evaluate_with_gpt(segments, student_speaker, course_level)

    if results:
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)

        print(f"\n📚 Course Level: {results.get('course_level', course_level)}")
        print(f"👤 Student Speaker: {results.get('student_speaker', student_speaker)}")

        print("\n📊 SCORES (1.0-5.0):")
        print("-" * 30)

        print("Content Relevance:")
        print(f"  • Task Coverage: {results.get('task_coverage', 'N/A')}")
        print(f"  • Appropriateness: {results.get('appropriateness', 'N/A')}")

        print("\nAccuracy:")
        print(f"  • Grammar Control: {results.get('grammar_control', 'N/A')}")
        print(f"  • Vocabulary Use: {results.get('vocabulary_use', 'N/A')}")

        print("\nCoherence:")
        print(f"  • Logical Flow: {results.get('logical_flow', 'N/A')}")
        print(f"  • Cohesive Devices: {results.get('cohesive_devices', 'N/A')}")

        print("\nDelivery (Estimated):")
        print(f"  • Pronunciation: {results.get('pronunciation', 'N/A')}")
        print(f"  • Intonation & Stress: {results.get('intonation_stress', 'N/A')}")

        print("-" * 30)
        print(f"\n⭐ AVERAGE SCORE: {results.get('average_score', 'N/A')}")

        # Vocabulary used
        if results.get('vocab_phrases_used'):
            print("\n📝 Key Vocabulary/Phrases Used:")
            for phrase in results['vocab_phrases_used'][:10]:
                print(f"  • {phrase}")

        # Feedback
        if results.get('feedback'):
            print("\n💬 Feedback:")
            print(f"  {results['feedback']}")

        # Improvement suggestions
        if results.get('improvement_suggestions'):
            print("\n🎯 Improvement Suggestions:")
            for i, suggestion in enumerate(results['improvement_suggestions'], 1):
                print(f"  {i}. {suggestion}")

        # Pronunciation analysis
        if results.get('pronunciation_analysis'):
            pa = results['pronunciation_analysis']
            print(f"\n🗣️ Pronunciation Analysis:")
            print(f"  • Error Rate: {pa['error_rate']:.1%}")
            print(f"  • Total Words: {pa['total_words']}")

            if pa.get('issues_found'):
                if show_detailed:
                    # Show ALL instances
                    print("\n  📋 ALL Pronunciation Issues (Detailed View):")
                    print("  " + "="*60)

                    for issue_type, examples in pa['issues_found'].items():
                        if examples:
                            display_name = issue_type.replace('_', '/').replace('/consonants', ' consonants')
                            print(f"\n  [{display_name.upper()}] - Total: {len(examples)} instances")
                            print("  " + "-"*55)

                            for idx, ex in enumerate(examples, 1):
                                transcribed = ex.get('transcribed', '')
                                intended = ex.get('likely_intended', '')
                                segment = ex.get('segment', '')

                                print(f"\n  {idx}. '{transcribed}' → '{intended}'")
                                if segment:
                                    # Show full context
                                    print(f"     Context: \"{segment}\"")

                            print("  " + "-"*55)

                    print("\n  " + "="*60)

                else:
                    # Show summary only
                    print("\n  📋 Pronunciation Issues (Summary):")
                    print("  " + "="*45)

                    for issue_type, examples in pa['issues_found'].items():
                        if examples:
                            display_name = issue_type.replace('_', '/').replace('/consonants', ' consonants')
                            print(f"\n  [{display_name}] - {len(examples)} instances:")

                            # Group by likely intended word
                            word_groups = {}
                            for ex in examples:
                                intended = ex.get('likely_intended', 'unknown')
                                if intended not in word_groups:
                                    word_groups[intended] = []
                                word_groups[intended].append(ex)

                            # Show top 5 most frequent errors
                            sorted_groups = sorted(word_groups.items(), key=lambda x: len(x[1]), reverse=True)
                            for intended, group in sorted_groups[:5]:
                                transcribed_words = list(set([g['transcribed'] for g in group]))
                                print(f"    • '{intended}' → {transcribed_words[0]} ({len(group)}x)")

                            if len(sorted_groups) > 5:
                                remaining = sum(len(g) for _, g in sorted_groups[5:])
                                print(f"    ... and {remaining} more instances")

                    print("\n  " + "="*45)
                    print("\n  💡 Tip: Run with 'y' for detailed pronunciation errors to see all instances")

        # Save results
        output_file = srt_file.replace('.srt', '_evaluation.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print("\n" + "="*50)
        print(f"✅ Full results saved to: {output_file}")

    else:
        print("\n❌ Evaluation failed. Please check your API key and try again.")

if __name__ == "__main__":
    main()