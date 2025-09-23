from flask import Flask, render_template, request, jsonify
import os
import json
from werkzeug.utils import secure_filename
from datetime import datetime
import tempfile
from speech_test_ass_advanced import AssemblyAIAdvancedEvaluator

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Allowed audio extensions
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'mp4', 'm4a', 'aac', 'ogg', 'flac'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main evaluation page"""
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    """Handle audio file upload and evaluation"""
    try:
        # Get form data
        student_id = request.form.get('student_id')
        course_order = request.form.get('course_order')
        course_level = int(request.form.get('course_level', 2))

        # Check if file was uploaded
        if 'audio_file' not in request.files:
            return jsonify({'error': 'No audio file uploaded'}), 400

        file = request.files['audio_file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{student_id}_{course_order}_{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Initialize evaluator
            evaluator = AssemblyAIAdvancedEvaluator()

            # Perform evaluation (using default model, no LeMUR for speed)
            results = evaluator.comprehensive_evaluation(
                audio_path=filepath,
                course_level=course_level,
                model_type=None,  # Use default model
                use_lemur=False   # Disable LeMUR for faster processing
            )

            # Add metadata
            results['student_id'] = student_id
            results['course_order'] = course_order

            # Prepare response data
            response_data = {
                'success': True,
                'student_id': student_id,
                'course_order': course_order,
                'course_level': course_level,
                'scores': {
                    'task_coverage': results.get('task_coverage', 5.0),
                    'appropriateness': results.get('appropriateness', 5.0),
                    'grammar_control': results.get('grammar_control', 5.0),
                    'vocabulary_use': results.get('vocabulary_use', 5.0),
                    'logical_flow': results.get('logical_flow', 5.0),
                    'cohesive_devices': results.get('cohesive_devices', 5.0),
                    'pronunciation': results.get('pronunciation', 5.0),
                    'intonation_stress': results.get('intonation_stress', 5.0),
                },
                'average_score': results.get('average_score', 5.0),
                'feedback': results.get('feedback', ''),
                'vocab_phrases': results.get('vocab_phrases_used', []),
                'student_speaker': results.get('student_speaker', 'Unknown'),
                'student_text': results.get('detailed_analysis', {}).get('student_text', ''),
                'word_count': results.get('detailed_analysis', {}).get('student_word_count', 0),
                'clarity_ratio': results.get('detailed_analysis', {}).get('pronunciation_details', {}).get('clarity_ratio', 0.5),
                'confidence': results.get('detailed_analysis', {}).get('pronunciation_details', {}).get('average_confidence', 0.5)
            }

            # Save results to JSON file
            result_filename = f"{student_id}_{course_order}_{timestamp}_results.json"
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, indent=2, ensure_ascii=False)

            return jsonify(response_data)

        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)

    except Exception as e:
        print(f"Evaluation error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # Create templates folder if it doesn't exist
    os.makedirs('templates', exist_ok=True)

    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5001)