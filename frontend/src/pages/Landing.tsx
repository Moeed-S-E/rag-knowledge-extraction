import { useEffect, useRef, useState } from 'react';
import type { MouseEvent, ReactNode, } from 'react';
import { Link } from 'react-router-dom';
import ThemeToggle from '../components/ThemeToggle';
import './Landing.css';

const HEADLINE = 'Answers you can check against the source.';
const QUESTION = 'Why do RNNs struggle with long sequences?';
const ANSWER =
    'Gradients shrink as they pass back through many time steps, so early inputs barely influence learning [1] Gated units were designed to reduce this [2]';

const STEPS = [
    { title: 'Retrieve', body: 'Your question is matched against the knowledge base and the closest passages are pulled. Add an entity boost to favor a specific term.' },
    { title: 'Answer', body: 'The answer is written from those passages only, with citation markers that point back to the chunk it came from.' },
    { title: 'Verify', body: 'A second pass checks whether the answer is supported. If it is not, you get a warning with the reason instead of a confident guess.' },
];

const prefersReduced = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/* Fades content in when it scrolls into view */
function Reveal({ children, delay = 0, className = '' }: { children: ReactNode; delay?: number; className?: string }) {
    const ref = useRef<HTMLDivElement>(null);
    const [shown, setShown] = useState(false);

    useEffect(() => {
        const el = ref.current;
        if (!el) return;
        const io = new IntersectionObserver(([e]) => {
            if (e.isIntersecting) { setShown(true); io.disconnect(); }
        }, { threshold: 0.2 });
        io.observe(el);
        return () => io.disconnect();
    }, []);

    return (
        <div ref={ref} className={`reveal ${shown ? 'reveal-in' : ''} ${className}`} style={{ transitionDelay: `${delay}ms` }}>
            {children}
        </div>
    );
}

/* Thin progress bar tied to page scroll */
function ScrollBar() {
    const ref = useRef<HTMLDivElement>(null);
    useEffect(() => {
        const onScroll = () => {
            const max = document.documentElement.scrollHeight - window.innerHeight;
            const p = max > 0 ? window.scrollY / max : 0;
            if (ref.current) ref.current.style.transform = `scaleX(${p})`;
        };
        onScroll();
        window.addEventListener('scroll', onScroll, { passive: true });
        return () => window.removeEventListener('scroll', onScroll);
    }, []);
    return <div ref={ref} className="scroll-bar" />;
}

/* Looping demo: types a question, searches, streams an answer, shows badges */
function LiveDemo() {
    const reduced = prefersReduced();
    const [q, setQ] = useState(reduced ? QUESTION : '');
    const [a, setA] = useState(reduced ? ANSWER : '');
    const [phase, setPhase] = useState<'idle' | 'search' | 'answer' | 'done'>(reduced ? 'done' : 'idle');

    useEffect(() => {
        if (reduced) return;
        let dead = false;
        const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));
        (async () => {
            while (!dead) {
                setQ(''); setA(''); setPhase('idle');
                await wait(700);
                for (let i = 1; i <= QUESTION.length && !dead; i++) { setQ(QUESTION.slice(0, i)); await wait(35); }
                await wait(350);
                setPhase('search');
                await wait(1500);
                setPhase('answer');
                for (let i = 1; i <= ANSWER.length && !dead; i++) { setA(ANSWER.slice(0, i)); await wait(18); }
                setPhase('done');
                await wait(5000);
            }
        })();
        return () => { dead = true; };
    }, [reduced]);

    const answerParts = a.replace(/\[\d*$/, '').split(/(\[\d+\])/g);

    return (
        <div className="surface-solid p-6 space-y-4 w-full" aria-label="Example answer, animated">
            <div className="text-xs opacity-50">Example</div>
            <div className="ml-auto w-fit max-w-[90%] rounded-2xl rounded-br-sm px-4 py-3 text-sm min-h-[2.75rem]" style={{ background: 'var(--surface-solid-hover)' }}>
                {q}
                {phase === 'idle' && <span className="blinking-caret" />}
            </div>

            <div className="min-h-[8.5rem] space-y-3">
                {phase === 'search' && (
                    <div>
                        <div className="text-sm font-medium opacity-80 mb-2"><span style={{ color: 'var(--mauve)' }}>✦</span> Searching sources…</div>
                        <div className="searching-bar" />
                    </div>
                )}
                {(phase === 'answer' || phase === 'done') && (
                    <p className={`text-[15px] leading-relaxed ${phase === 'answer' ? 'blinking-caret' : ''}`}>
                        {answerParts.map((part, i) =>
                            /^\[\d+\]$/.test(part) ? <span key={i} className="citation-chip pop">{part.slice(1, -1)}</span> : <span key={i}>{part}</span>,
                        )}
                    </p>
                )}
            </div>

            <div className="flex flex-wrap gap-2 min-h-[1.75rem]">
                {phase === 'done' && (
                    <>
                        <span className="glass-pill px-3 py-1 text-xs font-semibold pop" style={{ color: 'var(--mauve)' }}>Found 3 sources</span>
                        <span className="glass-pill px-3 py-1 text-xs font-semibold pop" style={{ color: 'var(--green)', animationDelay: '120ms' }}>✓ Verified</span>
                    </>
                )}
            </div>
        </div>
    );
}

/* Wraps the demo with a subtle cursor-driven 3D tilt and idle float */
function TiltCard({ children }: { children: ReactNode }) {
    const ref = useRef<HTMLDivElement>(null);
    const onMove = (e: MouseEvent) => {
        if (prefersReduced() || !ref.current) return;
        const r = ref.current.getBoundingClientRect();
        const x = (e.clientX - r.left) / r.width - 0.5;
        const y = (e.clientY - r.top) / r.height - 0.5;
        ref.current.style.transform = `rotateY(${x * 10}deg) rotateX(${-y * 10}deg)`;
    };
    const onLeave = () => { if (ref.current) ref.current.style.transform = ''; };

    return (
        <div style={{ perspective: 1000 }} onMouseMove={onMove} onMouseLeave={onLeave}>
            <div ref={ref} className="demo-tilt"><div className="float">{children}</div></div>
        </div>
    );
}

export default function Landing() {
    return (
        <div className="min-h-dvh">
            <ScrollBar />

            <div className="fixed inset-0 -z-10 pointer-events-none overflow-hidden" aria-hidden>
                <div className="orb orb-a" /><div className="orb orb-b" /><div className="orb orb-c" />
            </div>

            <nav className="glass-panel !rounded-full fixed top-4 inset-x-4 mx-auto max-w-5xl z-40 flex items-center justify-between py-2.5 pl-6 pr-3">
                <span className="font-semibold tracking-tight flex items-center gap-2">
                    <span style={{ color: 'var(--mauve)' }}>✦</span> Research Assistant
                </span>
                <div className="flex items-center gap-3">
                    <a href="#how" className="hidden sm:block text-sm font-medium opacity-70 hover:opacity-100 px-2">How it works</a>
                    <ThemeToggle />
                    <Link to="/chat" className="btn-primary btn-shine px-5 py-2 text-sm">Open chat</Link>
                </div>
            </nav>

            <header className="max-w-5xl mx-auto px-6 pt-32 pb-20 md:pt-40 grid md:grid-cols-[1.1fr_0.9fr] gap-14 items-center">
                <div>
                    <h1 className="text-5xl md:text-6xl font-semibold leading-[1.05] tracking-tight" aria-label={HEADLINE}>
                        {HEADLINE.split(' ').map((w, i) => (
                            <span key={i} className="word-in mr-[0.25em]" style={{ animationDelay: `${150 + i * 90}ms` }} aria-hidden>{w}</span>
                        ))}
                    </h1>
                    <Reveal delay={900}>
                        <p className="mt-6 text-lg max-w-md" style={{ color: 'var(--text-muted)' }}>
                            Ask your documents a question. Every answer shows the passages behind it and says so when it cannot back a claim up.
                        </p>
                    </Reveal>
                    <Reveal delay={1100} className="mt-9 flex flex-wrap gap-3">
                        <Link to="/chat" className="btn-primary btn-shine px-7 py-3">Start asking</Link>
                        <a href="#how" className="glass-pill px-7 py-3 font-semibold">See how it works</a>
                    </Reveal>
                </div>

                <Reveal delay={600}><TiltCard><LiveDemo /></TiltCard></Reveal>
            </header>

            <section id="how" className="max-w-5xl mx-auto px-6 pb-24 scroll-mt-24">
                <Reveal><h2 className="text-3xl font-semibold tracking-tight mb-8 max-w-md">How an answer gets built</h2></Reveal>
                <div className="glass-panel p-8 grid md:grid-cols-3 gap-8">
                    {STEPS.map((s, i) => (
                        <Reveal key={s.title} delay={i * 150}>
                            <div className="flex items-center gap-3 mb-3">
                                <span className="step-dot" style={{ animationDelay: `${i * 0.5}s` }} />
                                <h3 className="text-lg font-semibold">{s.title}</h3>
                            </div>
                            <p className="text-sm leading-relaxed" style={{ color: 'var(--text-muted)' }}>{s.body}</p>
                        </Reveal>
                    ))}
                </div>
            </section>

            <section className="max-w-5xl mx-auto px-6 pb-24 text-center">
                <Reveal>
                    <h2 className="text-3xl font-semibold tracking-tight mb-6">Ask your first question.</h2>
                    <Link to="/chat" className="btn-primary btn-shine px-8 py-3 inline-block">Open chat</Link>
                </Reveal>
            </section>
        </div>
    );
}