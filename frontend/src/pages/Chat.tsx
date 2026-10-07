import { useState } from 'react';
import type { FormEvent, ReactNode } from 'react';
import { Link } from 'react-router-dom';
import ThemeToggle from '../components/ThemeToggle';

const API_URL = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000';

const SUGGESTIONS = [
    'What is the Transformer architecture?',
    'Explain vanishing gradients',
    'How does Entity Boosting work?',
];

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
    };
}

/* ---------- small icons ---------- */
const Icon = ({ children, size = 12 }: { children: ReactNode; size?: number }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        {children}
    </svg>
);

/* ---------- sub-components ---------- */
function StatusChips({ r, onSources }: { r: QueryResponse; onSources: () => void }) {
    return (
        <div className="flex flex-wrap gap-2">
            <button type="button" onClick={onSources} className="glass-pill px-3 py-1 text-xs font-semibold flex items-center gap-1" style={{ color: 'var(--mauve)' }}>
                <Icon><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20" /></Icon>
                Found {r.retrieved_chunks} sources
            </button>
            {r.is_supported ? (
                <div className="glass-pill px-3 py-1 text-xs font-semibold flex items-center gap-1" style={{ color: 'var(--green)' }}>
                    <Icon><polyline points="20 6 9 17 4 12" /></Icon>
                    Verified
                </div>
            ) : (
                <div className="glass-pill px-3 py-1 text-xs font-semibold flex items-center gap-1" style={{ color: 'var(--orange)' }}>
                    <Icon><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" /><line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" /></Icon>
                    Potential hallucination
                </div>
            )}
        </div>
    );
}

function SourcesPanel({ citations, onClose }: { citations: string[]; onClose: () => void }) {
    return (
        <aside className="hidden lg:flex flex-col w-[320px] surface-solid !rounded-none shrink-0 z-10 animate-fade-up">
            <div className="p-6 pb-4 flex items-center justify-between">
                <h3 className="font-semibold">Sources ({citations.length})</h3>
                <button type="button" onClick={onClose} aria-label="Close sources" className="opacity-50 hover:opacity-100">
                    <Icon size={20}><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></Icon>
                </button>
            </div>
            <div className="flex-1 overflow-y-auto p-4 pt-0 space-y-4">
                {citations.map((c, i) => (
                    <div key={i} className="p-4 rounded-xl" style={{ backgroundColor: 'var(--surface-solid-hover)' }}>
                        <div className="text-xs font-semibold mb-2" style={{ color: 'var(--mauve)' }}>Chunk {i + 1}</div>
                        <div className="text-sm leading-relaxed opacity-90">{c}</div>
                    </div>
                ))}
            </div>
        </aside>
    );
}

/* ---------- page ---------- */
export default function Chat() {
    const [question, setQuestion] = useState('');
    const [asked, setAsked] = useState('');
    const [entityBoost, setEntityBoost] = useState('');
    const [loading, setLoading] = useState(false);
    const [response, setResponse] = useState<QueryResponse | null>(null);
    const [error, setError] = useState('');
    const [showSources, setShowSources] = useState(false);

    const reset = () => {
        setQuestion('');
        setAsked('');
        setResponse(null);
        setError('');
        setShowSources(false);
    };

    const submit = async (q: string) => {
        const text = q.trim();
        if (!text || loading) return;

        setAsked(text);
        setQuestion('');
        setLoading(true);
        setError('');
        setResponse(null);
        setShowSources(false);

        try {
            const res = await fetch(`${API_URL}/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: text, k: 3, entity_boost: entityBoost || null }),
            });
            if (!res.ok) throw new Error(`Server returned ${res.status}`);
            setResponse(await res.json());
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Something went wrong');
        } finally {
            setLoading(false);
        }
    };

    const onSubmit = (e: FormEvent) => {
        e.preventDefault();
        submit(question);
    };

    const onSuggestion = (q: string) => {
        setQuestion(q);
        submit(q);
    };

    const renderAnswer = (text: string) =>
        text.split(/(\[\d+\])/g).map((part, i) =>
            /^\[\d+\]$/.test(part) ? (
                <button key={i} type="button" className="citation-chip" title="View source" onClick={() => setShowSources(true)}>
                    {part.slice(1, -1)}
                </button>
            ) : (
                <span key={i}>{part}</span>
            ),
        );

    return (
        <div className="flex h-dvh overflow-hidden">
            {/* Sidebar */}
            <aside className="hidden md:flex flex-col w-[280px] glass-panel !rounded-none shrink-0 z-10">
                <div className="p-6">
                    <button type="button" onClick={reset} className="w-full flex items-center justify-center gap-2 py-3 rounded-full font-medium surface-solid !rounded-full">
                        <Icon size={16}><line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" /></Icon>
                        New chat
                    </button>
                </div>
                <div className="flex-1 overflow-y-auto p-6 pt-0">
                    <div className="text-sm font-semibold opacity-60 mb-4">Knowledge base</div>
                    <div className="text-sm opacity-70 italic">No custom documents loaded.</div>
                </div>
            </aside>

            {/* Main column */}
            <div className="flex-1 flex flex-col relative min-w-0">
                <header className="absolute top-4 inset-x-4 md:inset-x-8 glass-panel py-3 px-4 md:px-6 z-20 flex justify-between items-center gap-3">
                    <Link to="/" className="font-semibold tracking-tight text-lg flex items-center gap-2 min-w-0">
                        <span style={{ color: 'var(--mauve)' }}>✦</span>
                        <span className="truncate">Research Assistant</span>
                    </Link>
                    <div className="flex items-center gap-3 shrink-0">
                        <input
                            type="text"
                            aria-label="Entity boost"
                            placeholder="Entity boost (e.g. RNN)"
                            value={entityBoost}
                            onChange={(e) => setEntityBoost(e.target.value)}
                            className="glass-pill px-4 py-1.5 text-sm w-28 md:w-48 bg-transparent"
                        />
                        <ThemeToggle />
                    </div>
                </header>

                <main className="flex-1 overflow-y-auto pt-24 pb-32 px-4 md:px-0">
                    <div className="max-w-[760px] mx-auto w-full flex flex-col gap-8 min-h-full">
                        {!loading && !response && (
                            <div className="m-auto flex flex-col items-center text-center animate-fade-up">
                                <div className="w-12 h-12 rounded-full mb-6 flex items-center justify-center surface-solid !rounded-full">
                                    <span style={{ color: 'var(--mauve)', fontSize: 24 }}>✦</span>
                                </div>
                                <h2 className="text-2xl font-semibold mb-2">What do you want to look up?</h2>
                                <p className="opacity-60 mb-8 max-w-sm">Ask a question. You get an answer, the sources behind it, and a check for unsupported claims.</p>
                                <div className="flex flex-wrap justify-center gap-3">
                                    {SUGGESTIONS.map((q) => (
                                        <button key={q} type="button" onClick={() => onSuggestion(q)} className="surface-solid !rounded-full px-4 py-2 text-sm font-medium">
                                            {q}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        )}

                        {asked && (loading || response) && (
                            <div className="flex justify-end animate-fade-up">
                                <div className="surface-solid !rounded-br-sm px-6 py-4 max-w-[85%]">{asked}</div>
                            </div>
                        )}

                        {loading && (
                            <div className="flex justify-start animate-fade-up">
                                <div className="surface-solid !rounded-bl-sm px-6 py-5 w-full max-w-[85%]">
                                    <div className="flex items-center gap-3 mb-3">
                                        <span style={{ color: 'var(--mauve)' }}>✦</span>
                                        <span className="text-sm font-medium opacity-80">Searching sources…</span>
                                    </div>
                                    <div className="searching-bar" />
                                </div>
                            </div>
                        )}

                        {response && (
                            <div className="flex justify-start animate-fade-up">
                                <div className="surface-solid !rounded-bl-sm px-6 py-6 max-w-[85%] space-y-4">
                                    <StatusChips r={response} onSources={() => setShowSources((s) => !s)} />
                                    <div className="text-[16px] leading-[1.7]">{renderAnswer(response.answer)}</div>

                                    {!response.is_supported && (
                                        <div className="p-4 rounded-xl text-sm text-white" style={{ backgroundColor: 'var(--orange)' }}>
                                            <strong>Warning:</strong> {response.hallucination_reason}
                                        </div>
                                    )}

                                    <div className="pt-2 flex flex-wrap gap-x-4 text-xs opacity-50 font-mono">
                                        <span>Retrieval {response.latencies.retrieval_ms}ms</span>
                                        <span>Generation {response.latencies.generation_ms}ms</span>
                                        <span>Total {response.latencies.total_ms}ms</span>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </main>

                {/* Floating input */}
                <div className="absolute bottom-6 inset-x-4 md:inset-x-auto md:left-1/2 md:-translate-x-1/2 md:w-[760px] z-30">
                    {error && (
                        <div role="alert" className="absolute -top-14 left-1/2 -translate-x-1/2 toast-error text-sm animate-fade-up whitespace-nowrap">
                            {error}
                        </div>
                    )}
                    <form onSubmit={onSubmit} className="glass-pill p-2 flex items-center gap-2">
                        <input
                            type="text"
                            value={question}
                            onChange={(e) => setQuestion(e.target.value)}
                            placeholder="Ask a question…"
                            aria-label="Question"
                            className="flex-1 bg-transparent outline-none px-4 py-2 text-[16px]"
                            disabled={loading}
                            autoComplete="off"
                        />
                        <button type="submit" aria-label="Send" disabled={!question.trim() || loading} className="btn-primary w-10 h-10 flex items-center justify-center shrink-0 disabled:opacity-50">
                            <Icon size={18}><line x1="22" y1="2" x2="11" y2="13" /><polygon points="22 2 15 22 11 13 2 9 22 2" /></Icon>
                        </button>
                    </form>
                </div>
            </div>

            {showSources && response && <SourcesPanel citations={response.citations} onClose={() => setShowSources(false)} />}
        </div>
    );
}