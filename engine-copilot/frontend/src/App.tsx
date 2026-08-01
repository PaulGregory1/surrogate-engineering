import { useEffect } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import { TooltipProvider } from '@/components/ui/tooltip'
import { AppSidebar } from '@/components/AppSidebar'
import { ChatProvider, useChat } from '@/chat/ChatProvider'
import { ChatSidebar } from '@/chat/ChatSidebar'
import { Home } from '@/pages/Home'
import { FlowLab } from '@/pages/FlowLab'
import { StressLab } from '@/pages/StressLab'

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
    <div className="flex h-full overflow-hidden">
      <AppSidebar />
      <main className="flex min-w-0 flex-1 flex-col overflow-y-auto">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/flow" element={<FlowLab />} />
          <Route path="/stress" element={<StressLab />} />
          {/* The chat used to be its own page; it now lives in the sidebar on every
              page, so the old route just lands on the overview with the panel open. */}
          <Route path="/chat" element={<Navigate to="/" replace />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
      <ChatSidebar />
    </div>
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
