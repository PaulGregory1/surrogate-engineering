import { PanelRightOpen } from 'lucide-react'
import { useChat } from '@/chat/ChatProvider'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'

/**
 * The page's floating band: identity on the left, page-level controls on the right,
 * and the copilot re-open affordance — which only appears when the panel is closed,
 * so the header never shows a control for something already on screen.
 *
 * Built as a thin calque — .70 paper over a 16px blur — rather than an opaque bar,
 * so the drifting masses behind it stay visible as the page scrolls under it. It
 * spans the full column and keeps square corners, so it takes the band's opacity
 * by hand instead of the rounded `.cq-calque` class.
 */
export function PageHeader({
  title,
  subtitle,
  actions,
}: {
  title: string
  subtitle?: string
  actions?: React.ReactNode
}) {
  const { open, setOpen } = useChat()

  return (
    <header className="sticky top-0 z-20 flex h-14 shrink-0 items-center gap-4 border-b border-line bg-white/70 px-6 backdrop-blur-[16px] backdrop-saturate-150">
      <div className="min-w-0">
        {/* A page title is a panel title: 14px, weight 600. It is not a display
            slot — the 40px serif is reserved for a measured value. */}
        <h1 className="cq-title truncate">{title}</h1>
        {subtitle && <p className="cq-muted mt-0.5 truncate text-band">{subtitle}</p>}
      </div>
      <div className="ml-auto flex items-center gap-2">
        {actions}
        {!open && (
          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="outline" size="sm" onClick={() => setOpen(true)}>
                <PanelRightOpen />
                Copilot
              </Button>
            </TooltipTrigger>
            <TooltipContent>Open the copilot ⌘J</TooltipContent>
          </Tooltip>
        )}
      </div>
    </header>
  )
}
