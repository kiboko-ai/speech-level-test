import sqlite3
import json
from datetime import datetime
from pathlib import Path

class EvaluationDatabase:
    def __init__(self, db_path='evaluations.db'):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create students table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create evaluations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                course_order TEXT NOT NULL,
                course_level INTEGER NOT NULL,
                evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                -- Individual scores (10-point scale)
                task_coverage REAL,
                appropriateness REAL,
                grammar_control REAL,
                vocabulary_use REAL,
                logical_flow REAL,
                cohesive_devices REAL,
                pronunciation REAL,
                intonation_stress REAL,

                -- Overall score
                average_score REAL,

                -- Additional data
                feedback TEXT,
                vocab_phrases TEXT,  -- JSON array
                student_speaker TEXT,
                student_text TEXT,
                word_count INTEGER,
                clarity_ratio REAL,
                confidence REAL,

                -- Audio file info
                audio_filename TEXT,

                -- Progress tracking (JSON)
                progress_comparison TEXT,

                FOREIGN KEY (student_id) REFERENCES students(student_id)
            )
        ''')

        # Create index for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_student_evaluations
            ON evaluations(student_id, course_order)
        ''')

        conn.commit()
        conn.close()

    def add_student(self, student_id, name=None):
        """Add a new student or update existing"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT OR REPLACE INTO students (student_id, name, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (student_id, name))

            conn.commit()
            return True
        except Exception as e:
            print(f"Error adding student: {e}")
            return False
        finally:
            conn.close()

    def save_evaluation(self, evaluation_data):
        """Save evaluation results to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Ensure student exists
            self.add_student(evaluation_data['student_id'])

            # Convert vocab phrases list to JSON string
            vocab_json = json.dumps(evaluation_data.get('vocab_phrases', []))

            # Convert progress comparison to JSON string
            progress_json = json.dumps(evaluation_data.get('progress_comparison', {})) if 'progress_comparison' in evaluation_data else None

            cursor.execute('''
                INSERT INTO evaluations (
                    student_id, course_order, course_level,
                    task_coverage, appropriateness, grammar_control, vocabulary_use,
                    logical_flow, cohesive_devices, pronunciation, intonation_stress,
                    average_score, feedback, vocab_phrases, student_speaker,
                    student_text, word_count, clarity_ratio, confidence,
                    audio_filename, progress_comparison, evaluation_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                evaluation_data['student_id'],
                evaluation_data['course_order'],
                evaluation_data['course_level'],
                evaluation_data['scores']['task_coverage'],
                evaluation_data['scores']['appropriateness'],
                evaluation_data['scores']['grammar_control'],
                evaluation_data['scores']['vocabulary_use'],
                evaluation_data['scores']['logical_flow'],
                evaluation_data['scores']['cohesive_devices'],
                evaluation_data['scores']['pronunciation'],
                evaluation_data['scores']['intonation_stress'],
                evaluation_data['average_score'],
                evaluation_data.get('feedback', ''),
                vocab_json,
                evaluation_data.get('student_speaker', ''),
                evaluation_data.get('student_text', ''),
                evaluation_data.get('word_count', 0),
                evaluation_data.get('clarity_ratio', 0),
                evaluation_data.get('confidence', 0),
                evaluation_data.get('audio_filename', ''),
                progress_json
            ))

            conn.commit()
            return cursor.lastrowid
        except Exception as e:
            print(f"Error saving evaluation: {e}")
            return None
        finally:
            conn.close()

    def get_student_evaluations(self, student_id):
        """Get all evaluations for a student"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        cursor = conn.cursor()

        try:
            cursor.execute('''
                SELECT * FROM evaluations
                WHERE student_id = ?
                ORDER BY course_order
            ''', (student_id,))

            rows = cursor.fetchall()
            evaluations = []

            for row in rows:
                eval_dict = dict(row)
                # Parse JSON fields
                eval_dict['vocab_phrases'] = json.loads(eval_dict.get('vocab_phrases', '[]'))
                if eval_dict.get('progress_comparison'):
                    eval_dict['progress_comparison'] = json.loads(eval_dict['progress_comparison'])
                evaluations.append(eval_dict)

            return evaluations
        except Exception as e:
            print(f"Error fetching evaluations: {e}")
            return []
        finally:
            conn.close()

    def get_evaluation_by_course(self, student_id, course_order):
        """Get specific evaluation for a student and course"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute('''
                SELECT * FROM evaluations
                WHERE student_id = ? AND course_order = ?
            ''', (student_id, course_order))

            row = cursor.fetchone()
            if row:
                eval_dict = dict(row)
                eval_dict['vocab_phrases'] = json.loads(eval_dict.get('vocab_phrases', '[]'))
                if eval_dict.get('progress_comparison'):
                    eval_dict['progress_comparison'] = json.loads(eval_dict['progress_comparison'])
                return eval_dict
            return None
        except Exception as e:
            print(f"Error fetching evaluation: {e}")
            return None
        finally:
            conn.close()

    def student_exists(self, student_id):
        """Check if a student exists in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                SELECT COUNT(*) FROM students WHERE student_id = ?
            ''', (student_id,))

            count = cursor.fetchone()[0]
            return count > 0
        except Exception as e:
            print(f"Error checking student: {e}")
            return False
        finally:
            conn.close()

    def get_student_progress_summary(self, student_id):
        """Get summary of student's progress across all courses"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Get all evaluations with just the needed fields
            cursor.execute('''
                SELECT course_order, course_level, average_score, evaluation_date
                FROM evaluations
                WHERE student_id = ?
                ORDER BY course_order
            ''', (student_id,))

            evaluations = []
            for row in cursor.fetchall():
                evaluations.append({
                    'course_order': row[0],
                    'course_level': row[1],
                    'average_score': row[2],
                    'evaluation_date': row[3]
                })

            # Get overall statistics
            cursor.execute('''
                SELECT
                    COUNT(*) as total_evaluations,
                    AVG(average_score) as overall_average,
                    MAX(average_score) as highest_score,
                    MIN(average_score) as lowest_score
                FROM evaluations
                WHERE student_id = ?
            ''', (student_id,))

            stats = cursor.fetchone()

            # Handle empty results properly
            if stats and stats[0] and stats[0] > 0:
                statistics = {
                    'total_evaluations': stats[0],
                    'overall_average': round(stats[1], 1) if stats[1] else 0,
                    'highest_score': round(stats[2], 1) if stats[2] else 0,
                    'lowest_score': round(stats[3], 1) if stats[3] else 0
                }
            else:
                statistics = {
                    'total_evaluations': 0,
                    'overall_average': 0,
                    'highest_score': 0,
                    'lowest_score': 0
                }

            return {
                'evaluations': evaluations,
                'statistics': statistics
            }
        except Exception as e:
            print(f"Error getting progress summary: {e}")
            return {'evaluations': [], 'statistics': {}}
        finally:
            conn.close()