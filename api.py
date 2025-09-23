from flask import Flask, request, jsonify
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
CORS(app, origins=['http://localhost:3000'])  # Allow React dev server

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

# API Routes

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'API server is running'})

@app.route('/api/students/check', methods=['POST'])
def check_student():
    """Check if student exists and get their progress"""
    data = request.get_json()
    student_id = data.get('student_id')

    if not student_id:
        return jsonify({'error': 'Student ID required'}), 400

    # Ensure student is registered in the database
    if not db.student_exists(student_id):
        db.add_student(student_id)

    # Get student progress summary
    progress = db.get_student_progress_summary(student_id)

    return jsonify({
        'success': True,
        'student_id': student_id,
        'exists': len(progress['evaluations']) > 0,
        'progress': progress
    })

@app.route('/api/students/<student_id>/evaluations', methods=['GET'])
def get_student_evaluations(student_id):
    """Get all evaluations for a student"""
    evaluations = db.get_student_evaluations(student_id)
    return jsonify({
        'success': True,
        'student_id': student_id,
        'evaluations': evaluations
    })

@app.route('/api/students/<student_id>/evaluations/<course_order>', methods=['GET'])
def get_evaluation_detail(student_id, course_order):
    """Get specific evaluation detail"""
    evaluation = db.get_evaluation_by_course(student_id, course_order)

    if not evaluation:
        return jsonify({'error': 'Evaluation not found'}), 404

    return jsonify({
        'success': True,
        'evaluation': evaluation
    })

@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    """Handle audio file upload and evaluation"""
    try:
        # Get form data
        student_id = request.form.get('student_id')
        course_order = request.form.get('course_order')
        course_level = int(request.form.get('course_level', 2))

        # Validate input
        if not student_id or not course_order:
            return jsonify({'error': 'Missing required fields'}), 400

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

            # Perform evaluation
            results = evaluator.comprehensive_evaluation(
                audio_path=filepath,
                course_level=course_level,
                model_type=None,  # Use default model
                use_lemur=False   # Disable LeMUR for speed
            )

            # Add metadata
            results['student_id'] = student_id
            results['course_order'] = course_order

            # Get the most recent previous evaluation for comparison
            previous_evaluations = db.get_student_evaluations(student_id)
            previous_eval = None
            comparison_results = None

            if previous_evaluations and len(previous_evaluations) > 0:
                # Sort by date and get the most recent one
                sorted_evals = sorted(previous_evaluations,
                                    key=lambda x: x.get('evaluation_date', ''),
                                    reverse=True)
                previous_eval = sorted_evals[0]

                # Perform comparison if previous evaluation exists
                logging.info(f"Found previous evaluation from {previous_eval.get('evaluation_date')}")

                # Create comparison data
                current_eval_data = {
                    'task_coverage': results.get('task_coverage', 5.0),
                    'appropriateness': results.get('appropriateness', 5.0),
                    'grammar_control': results.get('grammar_control', 5.0),
                    'vocabulary_use': results.get('vocabulary_use', 5.0),
                    'logical_flow': results.get('logical_flow', 5.0),
                    'cohesive_devices': results.get('cohesive_devices', 5.0),
                    'pronunciation': results.get('pronunciation', 5.0),
                    'intonation_stress': results.get('intonation_stress', 5.0),
                    'vocab_phrases': results.get('vocab_phrases_used', [])
                }

                previous_eval_data = {
                    'task_coverage': previous_eval.get('task_coverage', 5.0),
                    'appropriateness': previous_eval.get('appropriateness', 5.0),
                    'grammar_control': previous_eval.get('grammar_control', 5.0),
                    'vocabulary_use': previous_eval.get('vocabulary_use', 5.0),
                    'logical_flow': previous_eval.get('logical_flow', 5.0),
                    'cohesive_devices': previous_eval.get('cohesive_devices', 5.0),
                    'pronunciation': previous_eval.get('pronunciation', 5.0),
                    'intonation_stress': previous_eval.get('intonation_stress', 5.0),
                    'vocab_phrases': previous_eval.get('vocab_phrases', [])
                }

                # Use the evaluator to compare
                comparison_results = evaluator.compare_with_previous(
                    current_eval_data,
                    previous_eval_data,
                    course_level
                )

                if comparison_results:
                    logging.info(f"Progress comparison completed. Average change: {comparison_results.get('average_change', 0):.2f}")

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

            # Add comparison results if available
            if comparison_results:
                response_data['progress_comparison'] = {
                    'change_scores': {
                        'task_coverage': comparison_results.get('change_task_coverage', 0),
                        'appropriateness': comparison_results.get('change_appropriateness', 0),
                        'grammar_control': comparison_results.get('change_grammar_control', 0),
                        'vocabulary_use': comparison_results.get('change_vocabulary_use', 0),
                        'logical_flow': comparison_results.get('change_logical_flow', 0),
                        'cohesive_devices': comparison_results.get('change_cohesive_devices', 0),
                        'pronunciation': comparison_results.get('change_pronunciation', 0),
                        'intonation_stress': comparison_results.get('change_intonation_stress', 0),
                    },
                    'average_change': comparison_results.get('average_change', 0),
                    'new_vocab_phrases': comparison_results.get('new_vocab_phrases', []),
                    'progress_summary': comparison_results.get('progress_summary', ''),
                    'remaining_issues': comparison_results.get('remaining_issues', ''),
                    'previous_course': previous_eval.get('course_order', ''),
                    'previous_date': previous_eval.get('evaluation_date', '')
                }

            # Save to database
            response_data['audio_filename'] = os.path.basename(filepath)
            db.save_evaluation(response_data)

            return jsonify(response_data)

        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)

    except Exception as e:
        logging.error(f"Evaluation error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/students/<student_id>/progress', methods=['GET'])
def get_student_progress(student_id):
    """Get student progress summary"""
    progress = db.get_student_progress_summary(student_id)
    return jsonify({
        'success': True,
        'student_id': student_id,
        'progress': progress
    })

if __name__ == '__main__':
    # Run the API server on port 5001
    app.run(debug=True, host='0.0.0.0', port=5001)