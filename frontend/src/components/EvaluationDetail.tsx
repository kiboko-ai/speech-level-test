import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Container, Box, Paper, Typography, Card, CardContent,
  LinearProgress, Button, Chip, Divider, IconButton
} from '@mui/material';
import {
  ArrowBack, Print, Mic, Speed, Psychology,
  TrendingUp, TrendingDown, Remove
} from '@mui/icons-material';
import {
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell
} from 'recharts';
import { getEvaluationDetail, Evaluation } from '../services/api';
import toast from 'react-hot-toast';

const EvaluationDetail: React.FC = () => {
  const { studentId, courseOrder } = useParams<{ studentId: string; courseOrder: string }>();
  const navigate = useNavigate();
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchEvaluation();
  }, [studentId, courseOrder]);

  const fetchEvaluation = async () => {
    if (!studentId || !courseOrder) return;

    try {
      const response = await getEvaluationDetail(studentId, courseOrder);
      setEvaluation(response.evaluation);
    } catch (error) {
      toast.error('Failed to load evaluation');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Container sx={{ py: 4 }}>
        <LinearProgress />
        <Typography sx={{ mt: 2, textAlign: 'center' }}>Loading evaluation...</Typography>
      </Container>
    );
  }

  if (!evaluation) {
    return (
      <Container sx={{ py: 4 }}>
        <Typography>Evaluation not found</Typography>
        <Button onClick={() => navigate('/')}>Back to Dashboard</Button>
      </Container>
    );
  }

  const radarData = [
    { category: 'Task Coverage', score: evaluation.task_coverage },
    { category: 'Appropriateness', score: evaluation.appropriateness },
    { category: 'Grammar', score: evaluation.grammar_control },
    { category: 'Vocabulary', score: evaluation.vocabulary_use },
    { category: 'Logic Flow', score: evaluation.logical_flow },
    { category: 'Cohesion', score: evaluation.cohesive_devices },
    { category: 'Pronunciation', score: evaluation.pronunciation },
    { category: 'Intonation', score: evaluation.intonation_stress },
  ];

  const barData = [
    { name: 'Task Coverage', value: evaluation.task_coverage, category: 'Content' },
    { name: 'Appropriateness', value: evaluation.appropriateness, category: 'Content' },
    { name: 'Grammar Control', value: evaluation.grammar_control, category: 'Accuracy' },
    { name: 'Vocabulary Use', value: evaluation.vocabulary_use, category: 'Accuracy' },
    { name: 'Logical Flow', value: evaluation.logical_flow, category: 'Coherence' },
    { name: 'Cohesive Devices', value: evaluation.cohesive_devices, category: 'Coherence' },
    { name: 'Pronunciation', value: evaluation.pronunciation, category: 'Delivery' },
    { name: 'Intonation', value: evaluation.intonation_stress, category: 'Delivery' },
  ];

  const getScoreColor = (score: number) => {
    if (score >= 8) return 'success.main';
    if (score >= 6) return 'info.main';
    if (score >= 4) return 'warning.main';
    return 'error.main';
  };

  const getCategoryColor = (category: string) => {
    const colors: any = {
      Content: '#667eea',
      Accuracy: '#764ba2',
      Coherence: '#f093fb',
      Delivery: '#fda085',
    };
    return colors[category] || '#667eea';
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Button
          startIcon={<ArrowBack />}
          onClick={() => navigate('/')}
          sx={{ mb: 2 }}
        >
          Back to Dashboard
        </Button>
        <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
          Evaluation Details
        </Typography>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <Chip label={`Student: ${studentId}`} />
          <Chip label={`Course: ${courseOrder}`} />
          <Chip label={`Level ${evaluation.course_level}`} color="primary" />
          <Chip label={new Date(evaluation.evaluation_date).toLocaleDateString()} variant="outlined" />
        </Box>
      </Box>

      {/* Overall Score */}
      <Paper sx={{ p: 4, mb: 4, background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', color: 'white' }}>
        <Typography variant="h5" sx={{ mb: 2 }}>Overall Performance</Typography>
        <Typography variant="h1" sx={{ fontWeight: 700 }}>
          {evaluation.average_score.toFixed(1)}/10
        </Typography>
        <Typography variant="h6" sx={{ opacity: 0.9, mt: 1 }}>
          {evaluation.average_score >= 8 ? 'Excellent' :
           evaluation.average_score >= 6 ? 'Good' :
           evaluation.average_score >= 4 ? 'Fair' : 'Needs Improvement'}
        </Typography>
      </Paper>

      {/* Metrics */}
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: 'repeat(2, 1fr)', md: 'repeat(3, 1fr)' }, gap: 3, mb: 4 }}>
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <Mic sx={{ mr: 1, color: 'primary.main' }} />
              <Typography color="text.secondary">Words Spoken</Typography>
            </Box>
            <Typography variant="h4">{evaluation.word_count}</Typography>
          </CardContent>
        </Card>
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <Speed sx={{ mr: 1, color: 'secondary.main' }} />
              <Typography color="text.secondary">Clarity</Typography>
            </Box>
            <Typography variant="h4">{(evaluation.clarity_ratio * 100).toFixed(0)}%</Typography>
          </CardContent>
        </Card>
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <Psychology sx={{ mr: 1, color: 'success.main' }} />
              <Typography color="text.secondary">Confidence</Typography>
            </Box>
            <Typography variant="h4">{(evaluation.confidence * 100).toFixed(0)}%</Typography>
          </CardContent>
        </Card>
      </Box>

      {/* Charts and Detailed Scores */}
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' }, gap: 3, mb: 4 }}>
        <Paper sx={{ p: 3, display: 'flex', flexDirection: 'column' }}>
          <Typography variant="h6" sx={{ mb: 2 }}>Performance Radar</Typography>
          <Box sx={{ flex: 1, minHeight: 400 }}>
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radarData} margin={{ top: 20, right: 30, bottom: 20, left: 30 }}>
                <PolarGrid
                  gridType="polygon"
                  radialLines={true}
                  strokeDasharray="0"
                  stroke="#e0e0e0"
                  strokeWidth={0.5}
                />
                {/* Custom polygon grid line for 5-unit mark */}
                <PolarGrid
                  gridType="polygon"
                  radialLines={false}
                  strokeDasharray="0"
                  stroke="#999"
                  strokeWidth={1.5}
                  polarRadius={[5]}
                  opacity={0.5}
                />
                <PolarAngleAxis dataKey="category" tick={{ fontSize: 12 }} />
                <PolarRadiusAxis
                  domain={[0, 10]}
                  tickCount={11}
                  tick={false}
                  axisLine={false}
                />
                <Radar
                  name="Score"
                  dataKey="score"
                  stroke="#667eea"
                  fill="#667eea"
                  fillOpacity={0.6}
                />
                <Tooltip
                  formatter={(value: number) => `${value.toFixed(1)}/10`}
                  contentStyle={{
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    border: '1px solid #667eea',
                    borderRadius: '4px'
                  }}
                />
              </RadarChart>
            </ResponsiveContainer>
          </Box>
        </Paper>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" sx={{ mb: 3 }}>Detailed Scores</Typography>
          <Box sx={{ display: 'grid', gridTemplateColumns: '1fr', gap: 2 }}>
            {[
              { title: 'Content Relevance', items: [
                { label: 'Task Coverage', value: evaluation.task_coverage },
                { label: 'Appropriateness', value: evaluation.appropriateness }
              ]},
              { title: 'Accuracy', items: [
                { label: 'Grammar Control', value: evaluation.grammar_control },
                { label: 'Vocabulary Use', value: evaluation.vocabulary_use }
              ]},
              { title: 'Coherence', items: [
                { label: 'Logical Flow', value: evaluation.logical_flow },
                { label: 'Cohesive Devices', value: evaluation.cohesive_devices }
              ]},
              { title: 'Delivery', items: [
                { label: 'Pronunciation', value: evaluation.pronunciation },
                { label: 'Intonation & Stress', value: evaluation.intonation_stress }
              ]}
            ].map((category) => (
              <Box key={category.title} sx={{ mb: 2 }}>
                <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                  {category.title}
                </Typography>
                {category.items.map((item) => (
                  <Box key={item.label} sx={{ mb: 1 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                      <Typography variant="body2" sx={{ fontSize: '0.875rem' }}>{item.label}</Typography>
                      <Typography variant="body2" sx={{ fontWeight: 600, color: getScoreColor(item.value), fontSize: '0.875rem' }}>
                        {item.value.toFixed(1)}/10
                      </Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={item.value * 10}
                      sx={{ height: 6, borderRadius: 3 }}
                    />
                  </Box>
                ))}
              </Box>
            ))}
          </Box>
        </Paper>
      </Box>

      {/* Feedback */}
      {evaluation.feedback && (
        <Paper sx={{ p: 3, mb: 4, bgcolor: 'primary.50' }}>
          <Typography variant="h6" sx={{ mb: 2 }}>Feedback</Typography>
          <Typography>{evaluation.feedback}</Typography>
        </Paper>
      )}

      {/* Progress Comparison */}
      {evaluation.progress_comparison && (
        <Paper sx={{ p: 3, mb: 4, bgcolor: 'background.paper', border: '2px solid', borderColor: 'primary.main' }}>
          <Typography variant="h6" sx={{ mb: 3, display: 'flex', alignItems: 'center' }}>
            📊 Progress Comparison with Previous Lesson
            <Chip
              label={`vs ${evaluation.progress_comparison.previous_course}`}
              sx={{ ml: 2 }}
              size="small"
              color="primary"
            />
          </Typography>

          {/* Overall Change Indicator */}
          <Box sx={{ mb: 3, p: 2, bgcolor: evaluation.progress_comparison.average_change > 0 ? 'success.50' : evaluation.progress_comparison.average_change < 0 ? 'error.50' : 'grey.100', borderRadius: 2 }}>
            <Typography variant="h5" sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              {evaluation.progress_comparison.average_change > 0 ? (
                <>
                  <TrendingUp sx={{ mr: 1, color: 'success.main' }} />
                  <span style={{ color: 'green' }}>+{evaluation.progress_comparison.average_change.toFixed(2)} Improvement</span>
                </>
              ) : evaluation.progress_comparison.average_change < 0 ? (
                <>
                  <TrendingDown sx={{ mr: 1, color: 'error.main' }} />
                  <span style={{ color: 'red' }}>{evaluation.progress_comparison.average_change.toFixed(2)} Decline</span>
                </>
              ) : (
                <>
                  <Remove sx={{ mr: 1 }} />
                  <span>No Change</span>
                </>
              )}
            </Typography>
          </Box>

          {/* Change Scores Grid */}
          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' }, gap: 2, mb: 3 }}>
            {[
              { label: 'Task Coverage', value: evaluation.progress_comparison.change_scores.task_coverage },
              { label: 'Appropriateness', value: evaluation.progress_comparison.change_scores.appropriateness },
              { label: 'Grammar Control', value: evaluation.progress_comparison.change_scores.grammar_control },
              { label: 'Vocabulary Use', value: evaluation.progress_comparison.change_scores.vocabulary_use },
              { label: 'Logical Flow', value: evaluation.progress_comparison.change_scores.logical_flow },
              { label: 'Cohesive Devices', value: evaluation.progress_comparison.change_scores.cohesive_devices },
              { label: 'Pronunciation', value: evaluation.progress_comparison.change_scores.pronunciation },
              { label: 'Intonation & Stress', value: evaluation.progress_comparison.change_scores.intonation_stress }
            ].map((item) => (
              <Box key={item.label} sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', p: 1.5, bgcolor: 'grey.50', borderRadius: 1 }}>
                <Typography variant="body2">{item.label}</Typography>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  {item.value > 0 ? (
                    <TrendingUp sx={{ mr: 0.5, color: 'success.main', fontSize: 20 }} />
                  ) : item.value < 0 ? (
                    <TrendingDown sx={{ mr: 0.5, color: 'error.main', fontSize: 20 }} />
                  ) : (
                    <Remove sx={{ mr: 0.5, color: 'grey.500', fontSize: 20 }} />
                  )}
                  <Typography
                    variant="body2"
                    sx={{
                      fontWeight: 600,
                      color: item.value > 0 ? 'success.main' : item.value < 0 ? 'error.main' : 'text.secondary'
                    }}
                  >
                    {item.value > 0 ? '+' : ''}{item.value.toFixed(1)}
                  </Typography>
                </Box>
              </Box>
            ))}
          </Box>

          {/* Progress Summary */}
          {evaluation.progress_comparison.progress_summary && (
            <Box sx={{ mb: 2, p: 2, bgcolor: 'success.50', borderRadius: 1 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>✅ Progress Summary</Typography>
              <Typography variant="body2">{evaluation.progress_comparison.progress_summary}</Typography>
            </Box>
          )}

          {/* Remaining Issues */}
          {evaluation.progress_comparison.remaining_issues && (
            <Box sx={{ mb: 2, p: 2, bgcolor: 'warning.50', borderRadius: 1 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>⚠️ Areas for Improvement</Typography>
              <Typography variant="body2">{evaluation.progress_comparison.remaining_issues}</Typography>
            </Box>
          )}

          {/* New Vocabulary */}
          {evaluation.progress_comparison.new_vocab_phrases && evaluation.progress_comparison.new_vocab_phrases.length > 0 && (
            <Box sx={{ p: 2, bgcolor: 'info.50', borderRadius: 1 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>🆕 New Vocabulary This Lesson</Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {evaluation.progress_comparison.new_vocab_phrases.map((phrase, index) => (
                  <Chip
                    key={index}
                    label={phrase}
                    size="small"
                    sx={{ bgcolor: 'white' }}
                  />
                ))}
              </Box>
            </Box>
          )}
        </Paper>
      )}

      {/* Vocabulary */}
      {evaluation.vocab_phrases && evaluation.vocab_phrases.length > 0 && (
        <Paper sx={{ p: 3, mb: 4 }}>
          <Typography variant="h6" sx={{ mb: 2 }}>Key Vocabulary Used</Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {evaluation.vocab_phrases.map((phrase, index) => (
              <Chip key={index} label={phrase} variant="outlined" />
            ))}
          </Box>
        </Paper>
      )}

      {/* Actions */}
      <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
        <Button
          variant="outlined"
          startIcon={<Print />}
          onClick={() => window.print()}
        >
          Print Report
        </Button>
      </Box>
    </Container>
  );
};

export default EvaluationDetail;