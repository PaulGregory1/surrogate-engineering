import { useEffect, useState } from 'react'
import { NavLink } from 'react-router-dom'
import { LayoutDashboard, Wind } from 'lucide-react'
import { getHealth } from '@/api'
import { cn } from '@/lib/utils'

const NAV = [
  { to: '/', label: 'Overview', icon: LayoutDashboard, end: true },
  { to: '/flow', label: 'Flow Lab', icon: Wind, end: false },
]

/** Backend reachability + whether the local `claude` CLI the agent drives is present.
 *  Surfaced permanently rather than only failing at the first chat message, because a
 *  missing CLI is a setup problem, not a runtime one.
 *
 *  No green, no red: the charte has neither. Nominal is plain ink, anything that
 *  deviates is burnt — and the wording carries the state on its own. */
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
    checking: { text: 'CONNECTING…', dot: 'bg-ink3', tone: 'text-ink3' },
    ok: { text: 'BACKEND + AGENT READY', dot: 'bg-ink', tone: 'text-ink3' },
    degraded: { text: 'AGENT UNAVAILABLE', dot: 'bg-burnt', tone: 'text-burnt' },
    down: { text: 'BACKEND UNREACHABLE', dot: 'bg-burnt', tone: 'text-burnt' },
  }[state]

  return (
    <div className="flex items-center gap-2 px-3.5 py-3" title={copy.text}>
      <span className={cn('size-1.5 shrink-0 rounded-full', copy.dot)} />
      <span className={cn('cq-k hidden truncate md:inline', copy.tone)}>{copy.text}</span>
    </div>
  )
}

export function AppSidebar() {
  return (
    // Narrow viewports collapse this to an icon rail rather than hiding it: a demo
    // where the lab becomes unreachable below a breakpoint is worse than a cramped
    // one. Labels and the health caption drop out, the targets stay.
    <aside className="no-print flex w-14 shrink-0 flex-col border-r border-line bg-sidebar backdrop-blur-xl backdrop-saturate-150 md:w-[228px]">
      <div className="flex h-14 shrink-0 items-center gap-2.5 border-b border-line px-3.5 md:px-4">
        {/* No logotype: the wordmark is set in the interface face at title size,
            like everything else. A display face here is what tips a whole app into
            the poster register. */}
        <span className="cq-title leading-none">NC</span>
        <div className="hidden leading-tight md:block">
          <div className="cq-k">COPILOT</div>
          <div className="cq-muted mt-0.5 text-band">Engineering surrogates</div>
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
                // The active mark is a 2px orange rule on the left edge, not a filled
                // pill: the nav has no quantity to encode, so it gets no orange flat.
                // Labels in Instrument Sans 13px, never mono: mono is reserved for
                // data, and a nav item is not a measurement.
                'flex items-center gap-2.5 rounded-r-md border-l-2 px-2.5 py-2 text-ctl transition-colors duration-150',
                isActive
                  ? 'border-or bg-peach/45 font-medium text-ink'
                  : 'border-transparent text-ink2 hover:bg-peach/20 hover:text-ink',
              )
            }
          >
            {({ isActive }) => (
              <>
                <Icon className={cn('size-4 shrink-0', isActive ? 'text-burnt' : 'text-ink3')} />
                <span className="hidden md:inline">{label}</span>
              </>
            )}
          </NavLink>
        ))}
      </nav>

      <div className="border-t border-line">
        <HealthDot />
      </div>
    </aside>
  )
}
