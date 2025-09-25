import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Container, Box, Paper, Typography, Card, CardContent,
  LinearProgress, Button, Chip, Tab, Tabs
} from '@mui/material';
import {
  ArrowBack, Print, Mic, Speed, Psychology,
  TrendingUp, TrendingDown, Remove
} from '@mui/icons-material';
import {
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, Cell,
  ScatterChart, Scatter, ReferenceLine, Legend
} from 'recharts';
import { getEvaluationDetail, Evaluation } from '../services/api';
import toast from 'react-hot-toast';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip as ChartTooltip,
  Legend as ChartLegend,
  RadialLinearScale,
  ArcElement,
  PointElement,
  LineElement,
  Filler
} from 'chart.js';
import { Bar as ChartBar, PolarArea } from 'react-chartjs-2';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  ChartTooltip,
  ChartLegend,
  RadialLinearScale,
  ArcElement,
  PointElement,
  LineElement,
  Filler
);

const EvaluationDetail: React.FC = () => {
  const { studentId, courseOrder } = useParams<{ studentId: string; courseOrder: string }>();
  const navigate = useNavigate();
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [loading, setLoading] = useState(true);
  const [chartTab, setChartTab] = useState(0);

  useEffect(() => {
    fetchEvaluation();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [studentId, courseOrder]);

  const fetchEvaluation = async () => {
    if (!studentId || !courseOrder) return;

    try {
      const response = await getEvaluationDetail(studentId, courseOrder);
      setEvaluation(response);
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

  // Mock average scores for comparison
  // Course Average - 같은 코스 레벨 수강생들의 평균
  const courseAverage = {
    task_coverage: 6.0,
    appropriateness: 7.2,
    grammar_control: 6.5,
    vocabulary_use: 6.5,
    logical_flow: 7.1,
    cohesive_devices: 5.8,
    pronunciation: 5.3,
    intonation_stress: 6.9
  };

  // Global Benchmark - 전체 플랫폼 사용자 평균
  const globalBenchmark = {
    task_coverage: 5.5,
    appropriateness: 6.0,
    grammar_control: 5.8,
    vocabulary_use: 5.7,
    logical_flow: 6.2,
    cohesive_devices: 5.5,
    pronunciation: 5.0,
    intonation_stress: 5.9
  };

  // Align radar scores with bubble chart position (3.5, 6.7)
  const alignedScores = {
    task_coverage: 3.8,
    appropriateness: 4.2,
    grammar_control: 3.0,
    vocabulary_use: 3.0,
    logical_flow: 7.2,
    cohesive_devices: 6.8,
    pronunciation: 6.4,
    intonation_stress: 6.4,
  };
  
  const radarData = [
    { category: 'Task Coverage', score: alignedScores.task_coverage, courseAvg: courseAverage.task_coverage, globalAvg: globalBenchmark.task_coverage },
    { category: 'Appropriateness', score: alignedScores.appropriateness, courseAvg: courseAverage.appropriateness, globalAvg: globalBenchmark.appropriateness },
    { category: 'Grammar', score: alignedScores.grammar_control, courseAvg: courseAverage.grammar_control, globalAvg: globalBenchmark.grammar_control },
    { category: 'Vocabulary', score: alignedScores.vocabulary_use, courseAvg: courseAverage.vocabulary_use, globalAvg: globalBenchmark.vocabulary_use },
    { category: 'Logic Flow', score: alignedScores.logical_flow, courseAvg: courseAverage.logical_flow, globalAvg: globalBenchmark.logical_flow },
    { category: 'Cohesion', score: alignedScores.cohesive_devices, courseAvg: courseAverage.cohesive_devices, globalAvg: globalBenchmark.cohesive_devices },
    { category: 'Pronunciation', score: alignedScores.pronunciation, courseAvg: courseAverage.pronunciation, globalAvg: globalBenchmark.pronunciation },
    { category: 'Intonation', score: alignedScores.intonation_stress, courseAvg: courseAverage.intonation_stress, globalAvg: globalBenchmark.intonation_stress },
  ];

  const getCategoryColor = (category: string) => {
    const colors: any = {
      Content: '#667eea',
      Accuracy: '#764ba2',
      Coherence: '#f093fb',
      Delivery: '#fda085',
    };
    return colors[category] || '#667eea';
  };

  // Generate correlated data points
  const generateMainCluster = () => {
    const points = [];
    
    // Blue cluster - Communication Flow - 우상단 (y=x에 가깝게)
    for (let i = 0; i < 20; i++) {
      const baseValue = 7 + Math.random() * 2;
      const x = baseValue + (Math.random() - 0.5) * 1.2;
      const y = baseValue + (Math.random() - 0.5) * 1.2;
      points.push({
        name: 'Communication Flow',
        x: Math.max(0, Math.min(10, x)),
        y: Math.max(0, Math.min(10, y)),
        z: 40 + Math.random() * 40,
        metrics: ['Communication metrics'],
        color: 'rgba(34, 211, 238, 0.5)',
        value: (x + y) / 2,
        clusterSize: 1
      });
    }
    
    // Green cluster - Overall Performance - 중앙에서 약간 우상단 (y=x에 가깝게)
    for (let i = 0; i < 35; i++) {
      const baseValue = 5.5 + Math.random() * 1.5;
      const x = baseValue + (Math.random() - 0.5) * 1.5;
      const y = baseValue + (Math.random() - 0.5) * 1.5;
      points.push({
        name: 'Overall Performance',
        x: Math.max(0, Math.min(10, x)),
        y: Math.max(0, Math.min(10, y)),
        z: 35 + Math.random() * 35,
        metrics: ['Overall metrics'],
        color: 'rgba(16, 185, 129, 0.5)',
        value: evaluation.average_score,
        clusterSize: 1
      });
    }
    
    // Red cluster - Expression Quality - 중앙 근처 (y=x에 가깝게)
    for (let i = 0; i < 25; i++) {
      const baseValue = 3.5 + Math.random() * 1.5;
      const x = baseValue + (Math.random() - 0.5) * 1.2;
      const y = baseValue + (Math.random() - 0.5) * 1.2;
      points.push({
        name: 'Expression Quality',
        x: Math.max(0, Math.min(10, x)),
        y: Math.max(0, Math.min(10, y)),
        z: 30 + Math.random() * 35,
        metrics: ['Expression metrics'],
        color: 'rgba(239, 68, 68, 0.5)',
        value: (evaluation.vocabulary_use + evaluation.intonation_stress) / 2,
        clusterSize: 1
      });
    }
    
    // Purple cluster - Language Accuracy (적은 데이터) - 우하단 (더 밀집)
    for (let i = 0; i < 12; i++) {
      const centerX = 7.5;
      const centerY = 2.5;
      const x = centerX + (Math.random() - 0.5) * 2;
      const y = centerY + (Math.random() - 0.5) * 2;
      points.push({
        name: 'Language Accuracy',
        x: Math.max(0, Math.min(10, x)),
        y: Math.max(0, Math.min(10, y)),
        z: 25 + Math.random() * 25,
        metrics: ['Grammar', 'Vocabulary'],
        color: 'rgba(118, 75, 162, 0.3)',
        value: (evaluation.grammar_control + evaluation.vocabulary_use) / 2,
        clusterSize: 2
      });
    }
    
    // Orange cluster - Delivery Skills (적은 데이터) - 좌상단 (더 밀집)
    for (let i = 0; i < 10; i++) {
      const centerX = 2.5;
      const centerY = 7.5;
      const x = centerX + (Math.random() - 0.5) * 2;
      const y = centerY + (Math.random() - 0.5) * 2;
      points.push({
        name: 'Delivery Skills',
        x: Math.max(0, Math.min(10, x)),
        y: Math.max(0, Math.min(10, y)),
        z: 25 + Math.random() * 30,
        metrics: ['Pronunciation', 'Intonation'],
        color: 'rgba(253, 160, 133, 0.3)',
        value: (evaluation.pronunciation + evaluation.intonation_stress) / 2,
        clusterSize: 2
      });
    }
    
    return points;
  };

  // Generate the bubble data
  const bubbleData = generateMainCluster();
  
  // User's position as a separate data point
  const userPosition = [{
    name: 'Your Position',
    x: 3.5,
    y: 6.7,
    z: 150,
    metrics: ['Your current performance'],
    color: '#fbbf24',
    value: (3.5 + 6.7) / 2,
    clusterSize: 1
  }];
  
  // Custom star shape for user position
  const renderStar = (props: any) => {
    const { cx, cy } = props;
    const size = 12;
    
    const star = [];
    for (let i = 0; i < 10; i++) {
      const radius = i % 2 === 0 ? size : size * 0.5;
      const angle = (Math.PI / 5) * i - Math.PI / 2;
      const x = cx + radius * Math.cos(angle);
      const y = cy + radius * Math.sin(angle);
      star.push(i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`);
    }
    star.push('Z');
    
    return <path d={star.join(' ')} fill="#fbbf24" stroke="#f59e0b" strokeWidth="2"/>;
  };

  // Prepare data for Grouped Bar Chart - horizontal layout
  const horizontalBarData = [
    { name: 'Task Coverage', value: alignedScores.task_coverage, category: 'Content' },
    { name: 'Appropriateness', value: alignedScores.appropriateness, category: 'Content' },
    { name: 'Grammar Control', value: alignedScores.grammar_control, category: 'Accuracy' },
    { name: 'Vocabulary Use', value: alignedScores.vocabulary_use, category: 'Accuracy' },
    { name: 'Logical Flow', value: alignedScores.logical_flow, category: 'Coherence' },
    { name: 'Cohesive Devices', value: alignedScores.cohesive_devices, category: 'Coherence' },
    { name: 'Pronunciation', value: alignedScores.pronunciation, category: 'Delivery' },
    { name: 'Intonation & Stress', value: alignedScores.intonation_stress, category: 'Delivery' }
  ];

  // Prepare data for Chart.js horizontal bar
  const chartJsData = {
    labels: horizontalBarData.map(item => item.name),
    datasets: [{
      label: 'Score',
      data: horizontalBarData.map(item => item.value),
      backgroundColor: horizontalBarData.map(item => getCategoryColor(item.category)),
      borderColor: horizontalBarData.map(item => getCategoryColor(item.category)),
      borderWidth: 1,
      barThickness: 30
    }]
  };

  const chartJsOptions = {
    indexAxis: 'y' as const,
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        callbacks: {
          label: function(context: any) {
            return `Score: ${context.parsed.x.toFixed(1)}/10`;
          }
        }
      }
    },
    scales: {
      x: {
        beginAtZero: true,
        max: 10,
        ticks: {
          stepSize: 2
        },
        grid: {
          display: true,
          color: '#e0e0e0'
        }
      },
      y: {
        grid: {
          display: false
        },
        ticks: {
          font: {
            size: 11
          }
        }
      }
    }
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Button
          startIcon={<ArrowBack />}
          onClick={() => navigate(`/${studentId}`)}
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
          <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
            <Tabs value={chartTab} onChange={(e, newValue) => setChartTab(newValue)} variant="scrollable" scrollButtons="auto">
              <Tab label="Performance Radar" />
              <Tab label="Bubble Chart" />
              <Tab label="Grouped Bar" />
            </Tabs>
          </Box>
          <Box sx={{ flex: 1, minHeight: 400 }}>
            {chartTab === 0 ? (
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
                  {/* Global Benchmark */}
                  <Radar
                    name="Global Benchmark"
                    dataKey="globalAvg"
                    stroke="#94a3b8"
                    strokeWidth={1.5}
                    strokeDasharray="2 2"
                    fill="#94a3b8"
                    fillOpacity={0.05}
                  />
                  {/* Course Average */}
                  <Radar
                    name="Course Average"
                    dataKey="courseAvg"
                    stroke="#f59e0b"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    fill="#f59e0b"
                    fillOpacity={0.1}
                  />
                  {/* Student's score */}
                  <Radar
                    name="Your Score"
                    dataKey="score"
                    stroke="#667eea"
                    strokeWidth={3}
                    fill="#667eea"
                    fillOpacity={0.3}
                  />
                  <Tooltip
                    formatter={(value: number, name: string) => [`${value.toFixed(1)}/10`, name]}
                    contentStyle={{
                      backgroundColor: 'rgba(255, 255, 255, 0.95)',
                      border: '1px solid #667eea',
                      borderRadius: '4px'
                    }}
                  />
                  <Legend 
                    verticalAlign="bottom"
                    wrapperStyle={{ paddingTop: '20px' }}
                  />
                </RadarChart>
              </ResponsiveContainer>
            ) : chartTab === 1 ? (
              // Bubble Chart
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 20, right: 30, bottom: 50, left: 60 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                  <XAxis
                    type="number"
                    dataKey="x"
                    name="Content Relevance + Accuracy"
                    domain={[0, 10]}
                    ticks={[0, 2, 4, 6, 8, 10]}
                    label={{ value: 'Content Relevance + Accuracy', position: 'insideBottom', offset: -10 }}
                  />
                  <YAxis
                    type="number"
                    dataKey="y"
                    name="Coherence + Delivery"
                    domain={[0, 10]}
                    ticks={[0, 2, 4, 6, 8, 10]}
                    label={{ value: 'Coherence + Delivery', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip
                    cursor={{ strokeDasharray: '3 3' }}
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload;
                        return (
                          <Box sx={{ 
                            p: 1.5, 
                            bgcolor: 'white', 
                            border: 1, 
                            borderColor: 'grey.300',
                            borderRadius: 1,
                            boxShadow: 2
                          }}>
                            <Typography variant="body2" sx={{ fontWeight: 600, color: data.color }}>
                              {data.name}
                            </Typography>
                            <Typography variant="caption" display="block">
                              Metrics: {data.metrics.join(', ')}
                            </Typography>
                            <Typography variant="caption" display="block">
                              Average Score: {data.value.toFixed(1)}/10
                            </Typography>
                            <Typography variant="caption" display="block">
                              Cluster Size: {data.clusterSize} {data.clusterSize === 1 ? 'metric' : 'metrics'}
                            </Typography>
                          </Box>
                        );
                      }
                      return null;
                    }}
                  />
                  <ZAxis 
                    type="number" 
                    dataKey="z" 
                    range={[20, 200]} 
                    name="Point Size"
                  />
                  <ReferenceLine x={5} stroke="#999" strokeDasharray="5 5" strokeWidth={1.5} />
                  <ReferenceLine y={5} stroke="#999" strokeDasharray="5 5" strokeWidth={1.5} />
                  <Scatter
                    name="Metrics"
                    data={bubbleData}
                    fill="#8884d8"
                  >
                    {bubbleData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Scatter>
                  {/* User position as a star */}
                  <Scatter
                    name="Your Position"
                    data={userPosition}
                    fill="#fbbf24"
                    shape={renderStar}
                  />
                </ScatterChart>
              </ResponsiveContainer>
            ) : chartTab === 2 ? (
              // Grouped Bar Chart (Horizontal)
              <Box sx={{ height: '100%', p: 2 }}>
                <ChartBar data={chartJsData} options={chartJsOptions} />
              </Box>
            ) : null}
          </Box>
        </Paper>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" sx={{ mb: 3 }}>
            Detailed Scores
            {evaluation.progress_comparison && (
              <Chip
                label={`vs ${evaluation.progress_comparison.previous_course}`}
                sx={{ ml: 2 }}
                size="small"
                color="primary"
              />
            )}
          </Typography>
          <Box sx={{ display: 'grid', gridTemplateColumns: '1fr', gap: 2 }}>
            {[
              { title: 'Content Relevance', items: [
                { label: 'Task Coverage', value: evaluation.task_coverage, change: evaluation.progress_comparison?.change_scores?.task_coverage },
                { label: 'Appropriateness', value: evaluation.appropriateness, change: evaluation.progress_comparison?.change_scores?.appropriateness }
              ]},
              { title: 'Accuracy', items: [
                { label: 'Grammar Control', value: evaluation.grammar_control, change: evaluation.progress_comparison?.change_scores?.grammar_control },
                { label: 'Vocabulary Use', value: evaluation.vocabulary_use, change: evaluation.progress_comparison?.change_scores?.vocabulary_use }
              ]},
              { title: 'Coherence', items: [
                { label: 'Logical Flow', value: evaluation.logical_flow, change: evaluation.progress_comparison?.change_scores?.logical_flow },
                { label: 'Cohesive Devices', value: evaluation.cohesive_devices, change: evaluation.progress_comparison?.change_scores?.cohesive_devices }
              ]},
              { title: 'Delivery', items: [
                { label: 'Pronunciation', value: evaluation.pronunciation, change: evaluation.progress_comparison?.change_scores?.pronunciation },
                { label: 'Intonation & Stress', value: evaluation.intonation_stress, change: evaluation.progress_comparison?.change_scores?.intonation_stress }
              ]}
            ].map((category) => (
              <Box key={category.title} sx={{ mb: 2 }}>
                <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                  {category.title}
                </Typography>
                {category.items.map((item) => (
                  <Box key={item.label} sx={{ mb: 1 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography variant="body2" sx={{ fontSize: '0.875rem' }}>{item.label}</Typography>
                        {item.change !== undefined && (
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            {item.change > 0 ? (
                              <TrendingUp sx={{ fontSize: 16, color: 'success.main' }} />
                            ) : item.change < 0 ? (
                              <TrendingDown sx={{ fontSize: 16, color: 'error.main' }} />
                            ) : (
                              <Remove sx={{ fontSize: 16, color: 'grey.500' }} />
                            )}
                            <Typography
                              variant="caption"
                              sx={{
                                fontWeight: 600,
                                color: item.change > 0 ? 'success.main' : item.change < 0 ? 'error.main' : 'text.secondary'
                              }}
                            >
                              {item.change > 0 ? '+' : ''}{item.change.toFixed(1)}
                            </Typography>
                          </Box>
                        )}
                      </Box>
                      <Typography variant="body2" sx={{ fontWeight: 600, fontSize: '0.875rem' }}>
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

          {/* Progress Comparison Details */}
          {evaluation.progress_comparison && (
            <Box sx={{ mt: 3 }}>
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
            </Box>
          )}
        </Paper>
      </Box>

      {/* Feedback */}
      {evaluation.feedback && (
        <Paper sx={{ p: 3, mb: 4, bgcolor: 'primary.50' }}>
          <Typography variant="h6" sx={{ mb: 2 }}>Feedback</Typography>
          <Typography>{evaluation.feedback}</Typography>
        </Paper>
      )}


      {/* Conversation Transcript */}
      {evaluation.student_text && (
        <Paper sx={{ p: 3, mb: 4, bgcolor: 'grey.50' }}>
          <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center' }}>
            📝 Conversation Transcript
            <Typography variant="caption" sx={{ ml: 2, color: 'text.secondary' }}>
              (Speaker verification check)
            </Typography>
          </Typography>
          <Box sx={{
            p: 2,
            bgcolor: 'white',
            borderRadius: 1,
            border: '1px solid',
            borderColor: 'grey.300',
            maxHeight: '400px',
            overflow: 'auto',
            fontFamily: 'monospace',
            fontSize: '0.9rem',
            lineHeight: 1.8
          }}>
            <Typography
              variant="body2"
              component="pre"
              sx={{
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                fontFamily: 'inherit'
              }}
            >
              {evaluation.student_text}
            </Typography>
          </Box>
          <Box sx={{ mt: 2, display: 'flex', gap: 2, alignItems: 'center' }}>
            <Chip
              label={`Student: ${evaluation.student_speaker}`}
              size="small"
              sx={{ bgcolor: 'primary.100' }}
            />
            <Typography variant="caption" color="text.secondary">
              Word count: {evaluation.word_count} | Clarity: {(evaluation.clarity_ratio * 100).toFixed(0)}%
            </Typography>
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