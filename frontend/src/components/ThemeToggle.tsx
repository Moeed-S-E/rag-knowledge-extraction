import { useTheme } from '../hooks/useTheme';

export default function ThemeToggle() {
    const { theme, toggle } = useTheme();
    const next = theme === 'light' ? 'dark' : 'light';

    return (
        <button
            type="button"
            onClick={toggle}
            aria-label={`Switch to ${next} theme`}
            className="surface-solid !rounded-full w-9 h-9 flex items-center justify-center shrink-0"
        >
            {theme === 'light' ? '🌙' : '☀️'}
        </button>
    );
}