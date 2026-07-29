export function ThemeToggle({ theme, onToggle }: { theme: 'dark' | 'light'; onToggle: () => void }) {
  return (
    <button
      onClick={onToggle}
      className="nc-card flex h-8 w-8 items-center justify-center text-sm hover:brightness-125 transition"
      title={theme === 'dark' ? 'Passer en thème clair' : 'Passer en thème sombre'}
    >
      {theme === 'dark' ? '☀' : '☽'}
    </button>
  )
}
