import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Dashboard from './components/Dashboard';
import EvaluationDetail from './components/EvaluationDetail';
import { Toaster } from 'react-hot-toast';

const theme = createTheme({
  palette: {
    primary: {
      main: '#667eea',
      light: '#8b9dc3',
      dark: '#4c63b6',
    },
    secondary: {
      main: '#764ba2',
      light: '#9969c7',
      dark: '#5a3d7a',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 700,
    },
    h2: {
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 12,
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Toaster position="top-center" />
      <Router>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/:studentId" element={<Dashboard />} />
          <Route path="/evaluation/:studentId/:courseOrder" element={<EvaluationDetail />} />
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;
