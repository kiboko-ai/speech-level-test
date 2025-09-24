import axios from 'axios';

const API_BASE_URL = 'http://localhost:5001';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface StudentProgress {
  evaluations: Evaluation[];
  statistics: {
    total_evaluations: number;
    overall_average: number;
    highest_score: number;
    lowest_score: number;
  };
}

export interface ProgressComparison {
  change_scores: {
    task_coverage: number;
    appropriateness: number;
    grammar_control: number;
    vocabulary_use: number;
    logical_flow: number;
    cohesive_devices: number;
    pronunciation: number;
    intonation_stress: number;
  };
  average_change: number;
  new_vocab_phrases: string[];
  progress_summary: string;
  remaining_issues: string;
  previous_course: string;
  previous_date: string;
}

export interface Evaluation {
  id?: number;
  student_id: string;
  course_order: string;
  course_level: number;
  evaluation_date: string;
  task_coverage: number;
  appropriateness: number;
  grammar_control: number;
  vocabulary_use: number;
  logical_flow: number;
  cohesive_devices: number;
  pronunciation: number;
  intonation_stress: number;
  average_score: number;
  feedback: string;
  vocab_phrases: string[];
  student_speaker: string;
  student_text: string;
  word_count: number;
  clarity_ratio: number;
  confidence: number;
  progress_comparison?: ProgressComparison;
}

export const checkStudent = async (studentId: string) => {
  const response = await api.post('/check_student', { student_id: studentId });
  return response.data;
};

export const getStudentProgress = async (studentId: string) => {
  const response = await api.get(`/students/${studentId}/progress`);
  return response.data;
};

export const getEvaluationDetail = async (studentId: string, courseOrder: string) => {
  const response = await api.get(`/evaluation/${studentId}/${courseOrder}`);
  return response.data;
};

export const evaluateAudio = async (formData: FormData) => {
  const response = await axios.post(`${API_BASE_URL}/evaluate`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export default api;