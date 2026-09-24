import { useState, FormEvent } from 'react';
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';
import { 
  Container, Typography, TextField, Button, Box, 
  CircularProgress, Chip, Accordion, AccordionSummary, AccordionDetails
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import SendIcon from '@mui/icons-material/Send';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';

// Create a solid clay-compatible dark theme with strictly 3 colors
// Background: #0c1618
// Elevated: #152528
// Text/Accent: #fdfaed
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#fdfaed', // All accents and buttons cream
    },
    secondary: {
      main: '#152528', 
    },
    background: {
      default: '#0c1618', // Darkest background
      paper: '#152528',   // Card background
    },
    success: {
      main: '#fdfaed', // No green, just cream
    },
    error: {
      main: '#fdfaed', // No red, just cream
    },
    text: {
      primary: '#fdfaed',
      secondary: 'rgba(253, 250, 237, 0.6)', // Cream with opacity
    }
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h3: {
      fontWeight: 800,
      color: '#fdfaed',
      letterSpacing: '-0.02em',
    }
  },
  shape: {
    borderRadius: 20,
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          boxShadow: 'none', 
          backgroundColor: 'transparent', 
        }
      }
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          color: '#fdfaed',
        }
      }
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            backgroundColor: '#0c1618', // Inset darkest color
            borderRadius: '12px',
            boxShadow: 'inset 4px 4px 8px rgba(0,0,0,0.6), inset -4px -4px 8px rgba(253,250,237,0.02)',
            '& fieldset': {
              border: 'none',
            },
          }
        }
      }
    }
  }
});

interface QueryResponse {
  answer: string;
  citations: string[];
  is_supported: boolean;
  hallucination_reason: string;
  retrieved_chunks: number;
  latencies: {
    retrieval_ms: number;
    generation_ms: number;
    hallucination_ms: number;
    total_ms: number;
  }
}

export default function App() {
  const [question, setQuestion] = useState("");
  const [entityBoost, setEntityBoost] = useState("");
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [error, setError] = useState("");

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setError("");
    setResponse(null);

    try {
      const res = await fetch("http://127.0.0.1:8000/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          question,
          k: 3,
          entity_boost: entityBoost || null
        })
      });

      if (!res.ok) {
        throw new Error(`Server returned ${res.status}`);
      }

      const data = await res.json();
      setResponse(data);
    } catch (err: any) {
      setError(err.message || "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <div className="min-h-screen py-12 px-4 selection:bg-[#152528]" style={{ backgroundColor: '#0c1618' }}>
        <Container maxWidth="md">
          {/* Header */}
          <Box className="text-center mb-10 animate-fade-in">
            <AutoAwesomeIcon sx={{ fontSize: 48, color: '#fdfaed', mb: 2 }} className="animate-pulse" />
            <Typography variant="h3" component="h1" gutterBottom>
              RAG Knowledge Engine
            </Typography>
            <Typography variant="subtitle1" sx={{ color: 'rgba(253, 250, 237, 0.5)' }}>
              Ask questions to the vector database. Entity boosting supported.
            </Typography>
          </Box>

          {/* Search Box - Claymorphism */}
          <div className="clay-card p-6 mb-8">
            <form onSubmit={handleSubmit} className="flex flex-col gap-4">
              <TextField
                fullWidth
                placeholder="What would you like to know?"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                autoFocus
                autoComplete="off"
                disabled={loading}
              />
              
              <Box className="flex flex-col sm:flex-row gap-4 items-center justify-between">
                <TextField
                  size="small"
                  placeholder="Entity Boost (Optional)"
                  value={entityBoost}
                  onChange={(e) => setEntityBoost(e.target.value)}
                  autoComplete="off"
                  disabled={loading}
                  sx={{ minWidth: { xs: '100%', sm: 250 } }}
                />
                
                <Button 
                  type="submit" 
                  size="large"
                  disabled={!question.trim() || loading}
                  endIcon={loading ? <CircularProgress size={20} sx={{color: '#fdfaed'}} /> : <SendIcon />}
                  className="clay-btn"
                  sx={{ 
                    minWidth: { xs: '100%', sm: 150 },
                    color: '#fdfaed',
                    backgroundColor: 'transparent',
                    '&:hover': { backgroundColor: 'transparent' }
                  }}
                >
                  {loading ? 'Searching...' : 'Ask Engine'}
                </Button>
              </Box>
            </form>
          </div>

          {/* Error Message */}
          {error && (
            <div className="clay-card p-4 mb-6 flex items-center gap-3" style={{ boxShadow: 'inset 2px 2px 8px rgba(0,0,0,0.5)', backgroundColor: '#0c1618' }}>
              <WarningAmberIcon sx={{color: '#fdfaed'}} />
              <Typography sx={{color: '#fdfaed'}}>{error}</Typography>
            </div>
          )}

          {/* Results Area */}
          {response && (
            <Box className="animate-slide-up space-y-8 mt-12">
              
              {/* Main Answer - Claymorphism */}
              <div className="clay-card p-8">
                <Typography variant="h6" className="mb-6 font-medium leading-relaxed" sx={{color: '#fdfaed'}}>
                  {response.answer}
                </Typography>
                
                {response.citations.length > 0 && (
                  <Box className="flex items-center gap-2 mt-6 pt-6 border-t" style={{ borderColor: 'rgba(253, 250, 237, 0.1)' }}>
                    <Typography variant="body2" sx={{color: 'rgba(253, 250, 237, 0.6)'}}>Sources:</Typography>
                    {response.citations.map((c, i) => (
                      <Chip key={i} label={`Chunk ${c}`} size="small" sx={{
                        backgroundColor: '#0c1618',
                        color: '#fdfaed',
                        boxShadow: 'inset 2px 2px 4px rgba(0,0,0,0.6), inset -2px -2px 4px rgba(253,250,237,0.02)'
                      }} />
                    ))}
                  </Box>
                )}
              </div>

              {/* Hallucination Status - Claymorphism */}
              <div className="clay-card p-6" style={{ backgroundColor: '#0c1618', boxShadow: 'inset 4px 4px 8px rgba(0,0,0,0.5), inset -2px -2px 6px rgba(253,250,237,0.02)'}}>
                <Box className="flex items-start gap-4">
                  {response.is_supported ? (
                    <CheckCircleIcon sx={{color: '#fdfaed'}} className="mt-1" />
                  ) : (
                    <WarningAmberIcon sx={{color: '#fdfaed'}} className="mt-1" />
                  )}
                  <Box>
                    <Typography variant="subtitle1" className="font-semibold mb-1" sx={{color: '#fdfaed'}}>
                      {response.is_supported ? 'Verified by LLM-as-a-Judge' : 'Warning: Potential Hallucination Detected'}
                    </Typography>
                    <Typography variant="body2" sx={{color: 'rgba(253, 250, 237, 0.6)'}}>
                      {response.hallucination_reason}
                    </Typography>
                  </Box>
                </Box>
              </div>

              {/* Advanced Analytics */}
              <div className="clay-card px-4">
                <Accordion sx={{ background: 'transparent', boxShadow: 'none', '&:before': { display: 'none' } }}>
                  <AccordionSummary expandIcon={<ExpandMoreIcon sx={{color: '#fdfaed'}} />} className="!px-2">
                    <Typography variant="subtitle2" className="font-semibold" sx={{color: 'rgba(253, 250, 237, 0.7)'}}>
                      Telemetry & Analytics
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails className="!px-2 !pb-6 !pt-0">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mt-4">
                      {[
                        { label: "Retrieved Chunks", val: response.retrieved_chunks },
                        { label: "Retrieval Time", val: `${response.latencies.retrieval_ms}ms` },
                        { label: "Generation Time", val: `${response.latencies.generation_ms}ms` },
                        { label: "Total Latency", val: `${response.latencies.total_ms}ms` },
                      ].map((stat, i) => (
                        <div key={i} className="flex flex-col items-center justify-center p-4 rounded-2xl" style={{
                          backgroundColor: '#0c1618',
                          boxShadow: 'inset 4px 4px 8px rgba(0,0,0,0.6), inset -4px -4px 8px rgba(253,250,237,0.02)'
                        }}>
                          <Typography variant="caption" className="uppercase font-bold tracking-wider mb-2" sx={{color: 'rgba(253, 250, 237, 0.5)'}}>
                            {stat.label}
                          </Typography>
                          <Typography variant="h6" className="font-mono" sx={{color: '#fdfaed'}}>
                            {stat.val}
                          </Typography>
                        </div>
                      ))}
                    </div>
                  </AccordionDetails>
                </Accordion>
              </div>
              
            </Box>
          )}
        </Container>
      </div>
    </ThemeProvider>
  );
}
