#!/usr/bin/env python3
import sqlite3
import json

# Connect to database
conn = sqlite3.connect('evaluations.db')
cursor = conn.cursor()

print("Starting database migration...")

# Create new table without unique constraint
cursor.execute('''
    CREATE TABLE IF NOT EXISTS evaluations_new (
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

# Copy data from old table (add NULL for progress_comparison column)
cursor.execute('''
    INSERT INTO evaluations_new (
        id, student_id, course_order, course_level, evaluation_date,
        task_coverage, appropriateness, grammar_control, vocabulary_use,
        logical_flow, cohesive_devices, pronunciation, intonation_stress,
        average_score, feedback, vocab_phrases, student_speaker,
        student_text, word_count, clarity_ratio, confidence,
        audio_filename, progress_comparison
    )
    SELECT
        id, student_id, course_order, course_level, evaluation_date,
        task_coverage, appropriateness, grammar_control, vocabulary_use,
        logical_flow, cohesive_devices, pronunciation, intonation_stress,
        average_score, feedback, vocab_phrases, student_speaker,
        student_text, word_count, clarity_ratio, confidence,
        audio_filename, NULL
    FROM evaluations
''')

# Drop old table
cursor.execute('DROP TABLE evaluations')

# Rename new table
cursor.execute('ALTER TABLE evaluations_new RENAME TO evaluations')

# Recreate index
cursor.execute('''
    CREATE INDEX idx_student_evaluations
    ON evaluations(student_id, course_order)
''')

conn.commit()

# Verify migration
cursor.execute('SELECT COUNT(*) FROM evaluations')
count = cursor.fetchone()[0]
print(f"Migration complete! {count} evaluations migrated successfully.")

conn.close()