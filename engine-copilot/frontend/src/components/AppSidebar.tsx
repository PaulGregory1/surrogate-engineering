import { useEffect, useState } from 'react'
import { NavLink } from 'react-router-dom'
import { Grid2x2, LayoutDashboard, Wind } from 'lucide-react'
import { getHealth } from '@/api'
import { cn } from '@/lib/utils'

const NAV = [
  { to: '/', label: 'Vue d’ensemble', icon: LayoutDashboard, end: true },
  { to: '/flow', label: 'Flow Lab', icon: Wind, end: false },
  { to: '/stress', label: 'Stress Lab', icon: Grid2x2, end: false },
]

/** Backend reachability + whether the local `claude` CLI the agent drives is present.
 *  Surfaced permanently rather than only failing at the first chat message, because a
 *  missing CLI is a setup problem, not a runtime one. */
function HealthDot() {
  const [state, setState] = useState<'checking' | 'ok' | 'degraded' | 'down'>('checking')

  useEffect(() => {
    let cancelled = false
    getHealth()
      .then((h) => {
        if (!cancelled) setState(h.claude_cli_installed ? 'ok' : 'degraded')
      })
      .catch(() => {
        if (!cancelled) setState('down')
      })
    return () => {
      cancelled = true
    }
  }, [])

  const copy = {
    checking: { text: 'Connexion…', color: 'bg-muted-foreground' },
    ok: { text: 'Backend + agent prêts', color: 'bg-good' },
    degraded: { text: 'Agent indisponible (claude CLI)', color: 'bg-warning' },
    down: { text: 'Backend injoignable', color: 'bg-critical' },
  }[state]

  return (
    <div
      className="flex items-center gap-2 px-3 py-2 text-[11px] text-muted-foreground md:px-3"
      title={copy.text}
    >
      <span className={cn('size-1.5 shrink-0 rounded-full', copy.color)} />
      <span className="hidden truncate md:inline">{copy.text}</span>
    </div>
  )
}

export function AppSidebar() {
  return (
    // Narrow viewports collapse this to an icon rail rather than hiding it: a demo
    // where the two labs become unreachable below a breakpoint is worse than a
    // cramped one. Labels and the health caption drop out, the targets stay.
    <aside className="no-print flex w-14 shrink-0 flex-col border-r border-sidebar-border bg-sidebar md:w-[228px]">
      <div className="flex h-14 shrink-0 items-center gap-2.5 border-b border-sidebar-border px-3.5 md:px-4">
        <div className="flex size-7 shrink-0 items-center justify-center rounded-md bg-primary text-primary-foreground">
          <span className="text-[11px] font-bold tracking-tight">NC</span>
        </div>
        <div className="hidden leading-tight md:block">
          <div className="text-sm font-semibold">Copilot</div>
          <div className="text-[10px] text-muted-foreground">Surrogates d’ingénierie</div>
        </div>
      </div>

      <nav className="flex flex-1 flex-col gap-0.5 p-2">
        {NAV.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            title={label}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-2.5 rounded-md px-2.5 py-2 text-sm transition',
                isActive
                  ? 'bg-sidebar-accent font-medium text-sidebar-accent-foreground'
                  : 'text-muted-foreground hover:bg-sidebar-accent/50 hover:text-foreground',
              )
            }
          >
            {({ isActive }) => (
              <>
                <Icon className={cn('size-4 shrink-0', isActive && 'text-ring')} />
                <span className="hidden md:inline">{label}</span>
              </>
            )}
          </NavLink>
        ))}
      </nav>

      <div className="border-t border-sidebar-border">
        <HealthDot />
      </div>
    </aside>
  )
}
