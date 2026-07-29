import { useEffect, useState } from 'react'
import { NavLink, Route, Routes } from 'react-router-dom'
import { Home } from './pages/Home'
import { FlowLab } from './pages/FlowLab'
import { ComingSoon } from './pages/ComingSoon'
import { ThemeToggle } from './components/ThemeToggle'

const TABS = [
  { to: '/', label: 'Home', end: true },
  { to: '/flow', label: 'Flow Lab', end: false },
  { to: '/stress', label: 'Stress Lab', end: false },
  { to: '/chat', label: 'Chat', end: false },
]

function App() {
  const [theme, setTheme] = useState<'dark' | 'light'>('dark')

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

  return (
    <div className="min-h-full flex flex-col">
      <nav className="nc-card sticky top-0 z-10 flex items-center gap-1 px-4 py-2.5 rounded-none border-x-0 border-t-0">
        <span className="font-semibold text-sm mr-4">Engine-Copilot</span>
        {TABS.map((t) => (
          <NavLink
            key={t.to}
            to={t.to}
            end={t.end}
            className={({ isActive }) =>
              `px-3 py-1.5 rounded-md text-sm transition ${isActive ? 'font-medium' : 'hover:brightness-125'}`
            }
            style={({ isActive }) => ({
              color: isActive ? 'var(--accent)' : 'var(--text-muted)',
              background: isActive ? 'color-mix(in srgb, var(--accent) 12%, transparent)' : 'transparent',
            })}
          >
            {t.label}
          </NavLink>
        ))}
        <div className="ml-auto">
          <ThemeToggle theme={theme} onToggle={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))} />
        </div>
      </nav>

      <main className="flex-1">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/flow" element={<FlowLab />} />
          <Route
            path="/stress"
            element={
              <ComingSoon
                title="Stress Lab"
                milestone="M4 — pas encore construit"
                description="Prédiction de contrainte de von Mises sur une plaque perforée via le GNN, avec optimiseur de positions de trous. Prochaine étape du plan."
              />
            }
          />
          <Route
            path="/chat"
            element={
              <ComingSoon
                title="Chat Orchestrator"
                milestone="M5 — pas encore construit"
                description="Agent Claude (Claude Agent SDK, branché sur l'abonnement Claude Code local) qui orchestre les deux labs via tool use. Prochaine étape du plan."
              />
            }
          />
        </Routes>
      </main>
    </div>
  )
}

export default App
