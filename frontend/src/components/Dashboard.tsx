import React, { useState } from 'react';
import {
  Container, Box, Paper, TextField, Button, Typography, Card, CardContent,
  Dialog, DialogTitle, DialogContent, DialogActions, FormControl, InputLabel,
  Select, MenuItem, LinearProgress, Alert, Fab, IconButton, Chip, Stack
} from '@mui/material';
import {
  Search, Add, Assessment, TrendingUp, School, CheckCircle
} from '@mui/icons-material';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell
} from 'recharts';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import { checkStudent, evaluateAudio, StudentProgress } from '../services/api';

const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const [studentId, setStudentId] = useState('');
  const [studentData, setStudentData] = useState<StudentProgress | null>(null);
  const [loading, setLoading] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [courseOrder, setCourseOrder] = useState('');
  const [courseLevel, setCourseLevel] = useState(2);
  const [audioFile, setAudioFile] = useState<File | null>(null);

  const handleCheckStudent = async () => {
    if (!studentId.trim()) {
      toast.error('Please enter a student ID');
      return;
    }

    setLoading(true);
    try {
      const response = await checkStudent(studentId);
      setStudentData(response.progress);
      if (response.progress.evaluations.length === 0) {
        toast.success('New student registered!');
      } else {
        toast.success('Student data loaded successfully');
      }
    } catch (error) {
      toast.error('Failed to check student');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const handleEvaluate = async () => {
    if (!audioFile || !courseOrder.trim()) {
      toast.error('Please fill all required fields');
      return;
    }

    const formData = new FormData();
    formData.append('student_id', studentId);
    formData.append('course_order', courseOrder);
    formData.append('course_level', courseLevel.toString());
    formData.append('audio_file', audioFile);

    setEvaluating(true);
    setModalOpen(false);

    const loadingToast = toast.loading('Processing evaluation... This may take 30-60 seconds');

    try {
      const response = await evaluateAudio(formData);
      toast.dismiss(loadingToast);
      toast.success('Evaluation completed successfully!');

      // Refresh student data
      const refreshResponse = await checkStudent(studentId);
      setStudentData(refreshResponse.progress);
    } catch (error) {
      toast.dismiss(loadingToast);
      toast.error('Evaluation failed. Please try again.');
      console.error(error);
    } finally {
      setEvaluating(false);
      setCourseOrder('');
      setAudioFile(null);
    }
  };

  const getScoreColor = (score: number) => {
    const hue = (score / 10) * 120;
    return `hsl(${hue}, 100%, 40%)`;  // Darker green by reducing lightness from 50% to 35%
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 4, textAlign: 'center' }}>
        <Typography variant="h3" component="h1" sx={{ fontWeight: 700, mb: 1 }}>
          🎯 Speech Evaluation Dashboard
        </Typography>
        <Typography variant="h6" color="text.secondary">
          AI-Powered English Assessment System
        </Typography>
      </Box>

      {/* Student Search */}
      <Paper sx={{ p: 4, mb: 4 }}>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <TextField
            fullWidth
            label="Student ID"
            variant="outlined"
            value={studentId}
            onChange={(e) => setStudentId(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleCheckStudent()}
            disabled={loading}
            sx={{ flex: 1 }}
          />
          <Button
            variant="contained"
            size="large"
            onClick={handleCheckStudent}
            disabled={loading}
            startIcon={<Search />}
            sx={{ px: 4 }}
          >
            Check Student
          </Button>
        </Box>
      </Paper>

      {/* Loading State */}
      {loading && <LinearProgress sx={{ mb: 4 }} />}

      {/* Student Dashboard */}
      {studentData && (
        <>
          {/* Statistics Cards */}
          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(4, 1fr)' }, gap: 3, mb: 4 }}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Assessment color="primary" sx={{ mr: 2 }} />
                  <Typography color="text.secondary">Total Evaluations</Typography>
                </Box>
                <Typography variant="h3">{studentData.statistics.total_evaluations}</Typography>
              </CardContent>
            </Card>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <TrendingUp color="secondary" sx={{ mr: 2 }} />
                  <Typography color="text.secondary">Overall Average</Typography>
                </Box>
                <Typography variant="h3">{studentData.statistics.overall_average.toFixed(1)}</Typography>
              </CardContent>
            </Card>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <CheckCircle sx={{ color: 'success.main', mr: 2 }} />
                  <Typography color="text.secondary">Highest Score</Typography>
                </Box>
                <Typography variant="h3">{studentData.statistics.highest_score.toFixed(1)}</Typography>
              </CardContent>
            </Card>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <School sx={{ color: 'warning.main', mr: 2 }} />
                  <Typography color="text.secondary">Lowest Score</Typography>
                </Box>
                <Typography variant="h3">{studentData.statistics.lowest_score.toFixed(1)}</Typography>
              </CardContent>
            </Card>
          </Box>

          {/* Progress Chart */}
          <Paper sx={{ p: 3, mb: 4 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
              <Typography variant="h5">Course Progress Overview</Typography>
              <Button
                variant="contained"
                startIcon={<Add />}
                onClick={() => setModalOpen(true)}
                disabled={evaluating}
              >
                Add Evaluation
              </Button>
            </Box>

            {studentData.evaluations.length > 0 ? (
              <ResponsiveContainer width="100%" height={400}>
                <BarChart
                  data={[...studentData.evaluations].sort((a, b) => {
                    // Convert course_order to numbers for proper sorting
                    const numA = parseInt(a.course_order);
                    const numB = parseInt(b.course_order);
                    return numA - numB;
                  })}
                  barCategoryGap="20%"
                  margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="course_order" axisLine={{ strokeWidth: 1 }} tickLine={{ strokeWidth: 1 }} />
                  <YAxis domain={[0, 10]} ticks={[0, 2, 4, 6, 8, 10]} />
                  <Tooltip />
                  <Bar
                    dataKey="average_score"
                    onClick={(data: any) => navigate(`/evaluation/${studentId}/${data.course_order}`)}
                    style={{ cursor: 'pointer' }}
                    maxBarSize={80}
                  >
                    {studentData.evaluations.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={getScoreColor(entry.average_score)} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <Box sx={{ textAlign: 'center', py: 8 }}>
                <Typography variant="h6" color="text.secondary" sx={{ mb: 2 }}>
                  No evaluations yet
                </Typography>
                <Typography color="text.secondary">
                  Click "Add Evaluation" to add the first evaluation
                </Typography>
              </Box>
            )}
          </Paper>
        </>
      )}

      {/* Evaluation Modal */}
      <Dialog open={modalOpen} onClose={() => setModalOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Add Course Evaluation</DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 2, display: 'flex', flexDirection: 'column', gap: 3 }}>
            <TextField
              fullWidth
              label="Course Order"
              value={courseOrder}
              onChange={(e) => setCourseOrder(e.target.value)}
              placeholder="e.g., Week 1, Lesson 3"
            />
            <FormControl fullWidth>
              <InputLabel>Course Level</InputLabel>
              <Select
                value={courseLevel}
                onChange={(e) => setCourseLevel(Number(e.target.value))}
                label="Course Level"
              >
                <MenuItem value={1}>Level 1 - Beginner</MenuItem>
                <MenuItem value={2}>Level 2 - Elementary</MenuItem>
                <MenuItem value={3}>Level 3 - Intermediate</MenuItem>
                <MenuItem value={4}>Level 4 - Advanced</MenuItem>
              </Select>
            </FormControl>
            <Button
              variant="outlined"
              component="label"
              fullWidth
              sx={{ py: 2 }}
            >
              {audioFile ? audioFile.name : 'Upload Audio File'}
              <input
                type="file"
                hidden
                accept="audio/*,.m4a"
                onChange={(e) => setAudioFile(e.target.files?.[0] || null)}
              />
            </Button>
            {audioFile && (
              <Alert severity="success">
                File selected: {audioFile.name} ({(audioFile.size / 1024 / 1024).toFixed(2)} MB)
              </Alert>
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setModalOpen(false)}>Cancel</Button>
          <Button onClick={handleEvaluate} variant="contained" disabled={!audioFile || !courseOrder}>
            Evaluate
          </Button>
        </DialogActions>
      </Dialog>

      {/* Evaluating Overlay */}
      {evaluating && (
        <Box
          sx={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            bgcolor: 'rgba(0, 0, 0, 0.8)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9999,
          }}
        >
          <Paper sx={{ p: 4, textAlign: 'center' }}>
            <LinearProgress sx={{ mb: 3, width: 300 }} />
            <Typography variant="h6">Processing Evaluation...</Typography>
            <Typography color="text.secondary" sx={{ mt: 1 }}>
              This may take 30-60 seconds
            </Typography>
          </Paper>
        </Box>
      )}
    </Container>
  );
};

export default Dashboard;