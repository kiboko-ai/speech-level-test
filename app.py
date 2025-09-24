from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import json
from werkzeug.utils import secure_filename
from datetime import datetime
import tempfile
from speech_test_ass_advanced import AssemblyAIAdvancedEvaluator
from database import EvaluationDatabase
from pydub import AudioSegment
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Initialize database
db = EvaluationDatabase()

# Allowed audio extensions
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'mp4', 'm4a', 'aac', 'ogg', 'flac'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_mp3(input_path, output_path):
    """Convert audio file to MP3 format using pydub"""
    try:
        logging.info(f"Converting {input_path} to MP3...")
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="mp3", bitrate="192k")
        logging.info(f"Successfully converted to {output_path}")
        return True
    except Exception as e:
        logging.error(f"Error converting audio: {e}")
        return False

@app.route('/')
def index():
    """Render the main dashboard page"""
    return render_template('dashboard.html')

@app.route('/check_student', methods=['POST'])
def check_student():
    """Check if student exists and get their progress"""
    data = request.get_json()
    student_id = data.get('student_id')

    if not student_id:
        return jsonify({'error': 'Student ID required'}), 400

    # Get student progress summary
    progress = db.get_student_progress_summary(student_id)

    return jsonify({
        'success': True,
        'student_id': student_id,
        'exists': len(progress['evaluations']) > 0,
        'progress': progress
    })

@app.route('/evaluation/<student_id>/<course_order>')
def evaluation_detail(student_id, course_order):
    """Show detailed evaluation for a specific course"""
    evaluation = db.get_evaluation_by_course(student_id, course_order)

    if not evaluation:
        return "Evaluation not found", 404

    return render_template('evaluation_detail.html',
                         student_id=student_id,
                         course_order=course_order,
                         evaluation=evaluation)

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
        original_ext = filename.rsplit('.', 1)[1].lower()

        # Save the uploaded file
        temp_filename = f"{student_id}_{course_order}_{timestamp}_{filename}"
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(temp_filepath)

        # Check if conversion is needed
        filepath = temp_filepath
        if original_ext == 'm4a':
            # Convert m4a to mp3
            mp3_filename = temp_filename.rsplit('.', 1)[0] + '.mp3'
            mp3_filepath = os.path.join(app.config['UPLOAD_FOLDER'], mp3_filename)

            logging.info(f"Converting m4a file to mp3: {temp_filename} -> {mp3_filename}")

            if convert_to_mp3(temp_filepath, mp3_filepath):
                # Use the converted mp3 file
                filepath = mp3_filepath
                # Remove the original m4a file
                os.remove(temp_filepath)
                logging.info("Successfully converted m4a to mp3")
            else:
                # If conversion fails, try to use the original file anyway
                logging.warning("Failed to convert m4a to mp3, attempting with original file")

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

            # Save to database (use the actual filename that was processed)
            response_data['audio_filename'] = os.path.basename(filepath)
            db.save_evaluation(response_data)

            # Also save to JSON file for backup
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