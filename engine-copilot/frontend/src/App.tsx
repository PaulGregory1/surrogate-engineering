import { useEffect } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import { TooltipProvider } from '@/components/ui/tooltip'
import { AppSidebar } from '@/components/AppSidebar'
import { ContourField } from '@/components/ContourField'
import { ChatProvider, useChat } from '@/chat/ChatProvider'
import { ChatSidebar } from '@/chat/ChatSidebar'
import { Home } from '@/pages/Home'
import { FlowLab } from '@/pages/FlowLab'
import { StressLab } from '@/pages/StressLab'
import { BracketLab } from '@/pages/BracketLab'

/**
 * The viewport-wide background layers: the drifting gradient masses, then the grain.
 *
 * The charte's third layer — large hard-edged circles — is *not* here. Two arcs
 * sweeping across the whole window read as stray strokes rather than as structure,
 * so the job of holding the blur in place is done instead by `<ContourField>`, which
 * is both sharper and scoped to the content column.
 *
 * Everything here is `position: fixed` and `pointer-events: none`; the shell below
 * carries its own stacking context so panels always sit on top.
 */
function Backdrop() {
  return (
    <>
      <div className="cq-bg" aria-hidden>
        <div className="cq-blob cq-b1" />
        <div className="cq-blob cq-b2" />
        <div className="cq-blob cq-b3" />
      </div>
      {/* Three lines of turbulence that take the plastic look off the white. */}
      <svg className="cq-grain" aria-hidden>
        <filter id="cq-grain-filter">
          <feTurbulence type="fractalNoise" baseFrequency="0.85" numOctaves={4} />
          <feColorMatrix type="saturate" values="0" />
        </filter>
        <rect width="100%" height="100%" filter="url(#cq-grain-filter)" opacity="0.14" />
      </svg>
    </>
  )
}

function Shell() {
  const { toggle } = useChat()

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'j') {
        e.preventDefault()
        toggle()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [toggle])

  return (
    <>
      <Backdrop />
      <div className="relative z-10 flex h-full overflow-hidden">
        <AppSidebar />
        {/* The contour map belongs to the content column alone — never behind the
            nav rail or the copilot panel. It lives on this wrapper rather than
            inside the scroller, so it stays put while the page scrolls over it. */}
        <div className="relative flex min-w-0 flex-1 flex-col overflow-hidden">
          <ContourField />
          <main className="relative z-10 flex flex-1 flex-col overflow-y-auto">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/flow" element={<FlowLab />} />
              <Route path="/stress" element={<StressLab />} />
              <Route path="/bracket" element={<BracketLab />} />
              {/* The chat used to be its own page; it now lives in the sidebar on
                  every page, so the old route just lands on the overview with the
                  panel open. */}
              <Route path="/chat" element={<Navigate to="/" replace />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </main>
        </div>
        <ChatSidebar />
      </div>
    </>
  )
}

export default function App() {
  return (
    <ChatProvider>
      <TooltipProvider delayDuration={200}>
        <Shell />
      </TooltipProvider>
    </ChatProvider>
  )
}
