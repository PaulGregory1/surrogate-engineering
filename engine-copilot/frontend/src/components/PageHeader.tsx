import { PanelRightOpen } from 'lucide-react'
import { useChat } from '@/chat/ChatProvider'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'

/** Sticky page header: identity on the left, page-level controls on the right, and
 *  the copilot re-open affordance — which only appears when the panel is closed, so
 *  the header never shows a control for something already on screen. */
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
    <header className="sticky top-0 z-20 flex h-14 shrink-0 items-center gap-4 border-b border-border bg-background/80 px-6 backdrop-blur-md">
      <div className="min-w-0">
        <h1 className="truncate text-sm font-semibold">{title}</h1>
        {subtitle && <p className="truncate text-xs text-muted-foreground">{subtitle}</p>}
      </div>
      <div className="ml-auto flex items-center gap-2">
        {actions}
        {!open && (
          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="outline" size="sm" onClick={() => setOpen(true)}>
                <PanelRightOpen />
                Copilote
              </Button>
            </TooltipTrigger>
            <TooltipContent>Ouvrir le copilote ⌘J</TooltipContent>
          </Tooltip>
        )}
      </div>
    </header>
  )
}
