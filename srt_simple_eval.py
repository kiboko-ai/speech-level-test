import re
from pathlib import Path
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import Dict, List

load_dotenv()

class SimpleSRTEvaluator:
    def __init__(self):
        """Initialize with OpenAI client"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        self.client = OpenAI(api_key=api_key)

    def parse_srt(self, srt_path: str) -> List[Dict]:
        """Parse SRT file"""
        segments = []
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        blocks = content.strip().split('\n\n')
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                timestamp_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', lines[1])
                if timestamp_match:
                    text = ' '.join(lines[2:])

                    # Parse speaker
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
                        'text': clean_text
                    })
        return segments

    def evaluate_transcript(self, segments: List[Dict], student_speaker: str,
                           course_level: int = 2) -> Dict:
        """Evaluate using GPT-4 without pronunciation analysis"""

        # Extract student text
        student_segments = [seg for seg in segments if seg['speaker'] == student_speaker]
        student_text = " ".join([seg['text'] for seg in student_segments])

        # Create dialogue
        full_dialogue = "\n".join([f"{seg['speaker']}: {seg['text']}" for seg in segments])

        prompt = f"""You are an expert English speaking test evaluator.

Student's speech only:
{student_text}

Full dialogue for context:
{full_dialogue}

Course level: {course_level} (1-4 scale)
- Levels 1-2: Basic communication, vocabulary building
- Levels 3-4: Accuracy, complex structures, fluency

Evaluate ONLY what can be assessed from text (NOT pronunciation):

1. Content Relevance (1.0-5.0, increment 0.1)
   - task_coverage: Addressing topics/questions
   - appropriateness: Staying on-topic

2. Accuracy (1.0-5.0, increment 0.1)
   - grammar_control: Sentence structure, tenses
   - vocabulary_use: Range and precision

3. Coherence (1.0-5.0, increment 0.1)
   - logical_flow: Clear reasoning
   - cohesive_devices: Linking words

Identify:
- Key vocabulary used (TOEIC/OPIc relevant)
- Main strengths and weaknesses
- Specific suggestions

Output JSON:
{{
  "course_level": {course_level},
  "student_speaker": "{student_speaker}",
  "task_coverage": <score>,
  "appropriateness": <score>,
  "grammar_control": <score>,
  "vocabulary_use": <score>,
  "logical_flow": <score>,
  "cohesive_devices": <score>,
  "average_score": <calculated average of 6 scores>,
  "vocab_phrases_used": ["phrase1", "phrase2", ...],
  "strengths": ["strength1", "strength2"],
  "weaknesses": ["weakness1", "weakness2"],
  "suggestions": ["suggestion1", "suggestion2"],
  "detailed_feedback": "Comprehensive feedback paragraph"
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert English evaluator. Be accurate with scores."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            return json.loads(response.choices[0].message.content)

        except Exception as e:
            print(f"Error: {e}")
            return None

def main():
    print("="*60)
    print("   SRT-Based English Evaluation (Text Analysis Only)")
    print("="*60)

    # Default values
    default_srt = '/Users/jason/Downloads/35526.srt'

    # Get inputs
    srt_file = input(f"\nSRT file path (default: {default_srt}): ").strip() or default_srt

    if not Path(srt_file).exists():
        print(f"❌ File not found: {srt_file}")
        return

    course_level = input("Course level 1-4 (default: 2): ").strip()
    course_level = int(course_level) if course_level and course_level.isdigit() else 2

    student_choice = input("Student speaker A or B (default: B): ").strip().upper() or 'B'

    # Initialize evaluator
    evaluator = SimpleSRTEvaluator()

    print(f"\n📝 Processing: {Path(srt_file).name}")
    print("-" * 50)

    # Parse SRT
    segments = evaluator.parse_srt(srt_file)
    speakers = list(set(seg['speaker'] for seg in segments))
    print(f"✓ Found {len(segments)} segments")
    print(f"✓ Speakers: {', '.join(speakers)}")

    # Find student speaker
    student_speaker = f'Speaker{student_choice}'
    if student_speaker not in speakers:
        # Try alternatives
        for alt in [f'speaker{student_choice}', student_choice, f'SPEAKER{student_choice}']:
            if alt in speakers:
                student_speaker = alt
                break

    print(f"✓ Student: {student_speaker}")

    # Word count
    student_words = sum(len(seg['text'].split())
                       for seg in segments if seg['speaker'] == student_speaker)
    print(f"✓ Student words: {student_words}")

    # Evaluate
    print("\n⏳ Evaluating with GPT-4...")
    results = evaluator.evaluate_transcript(segments, student_speaker, course_level)

    if results:
        print("\n" + "="*60)
        print("                    EVALUATION RESULTS")
        print("="*60)

        print(f"\n📚 Course Level: {course_level}")
        print(f"👤 Student: {student_speaker}")
        print(f"💬 Word Count: {student_words}")

        print("\n" + "─"*50)
        print("                      SCORES")
        print("─"*50)

        print("\n📖 Content Relevance:")
        print(f"   Task Coverage:    {results.get('task_coverage', 'N/A')}")
        print(f"   Appropriateness:  {results.get('appropriateness', 'N/A')}")

        print("\n✏️ Accuracy:")
        print(f"   Grammar Control:  {results.get('grammar_control', 'N/A')}")
        print(f"   Vocabulary Use:   {results.get('vocabulary_use', 'N/A')}")

        print("\n🔗 Coherence:")
        print(f"   Logical Flow:     {results.get('logical_flow', 'N/A')}")
        print(f"   Cohesive Devices: {results.get('cohesive_devices', 'N/A')}")

        print("\n" + "─"*50)
        avg = results.get('average_score', 'N/A')
        print(f"⭐ AVERAGE SCORE: {avg}")
        print("─"*50)

        # Vocabulary
        if results.get('vocab_phrases_used'):
            print("\n📝 Key Vocabulary Used:")
            for phrase in results['vocab_phrases_used'][:8]:
                print(f"   • {phrase}")

        # Strengths & Weaknesses
        if results.get('strengths'):
            print("\n✅ Strengths:")
            for s in results['strengths']:
                print(f"   • {s}")

        if results.get('weaknesses'):
            print("\n⚠️ Areas to Improve:")
            for w in results['weaknesses']:
                print(f"   • {w}")

        # Suggestions
        if results.get('suggestions'):
            print("\n💡 Suggestions:")
            for i, suggestion in enumerate(results['suggestions'], 1):
                print(f"   {i}. {suggestion}")

        # Detailed feedback
        if results.get('detailed_feedback'):
            print("\n📋 Detailed Feedback:")
            print(f"   {results['detailed_feedback']}")

        # Save JSON
        output_file = srt_file.replace('.srt', '_simple_eval.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print("\n" + "="*60)
        print(f"💾 Results saved to: {output_file}")
        print("="*60)

if __name__ == "__main__":
    main()