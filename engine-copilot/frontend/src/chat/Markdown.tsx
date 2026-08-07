import ReactMarkdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'

/**
 * The agent answers in Markdown, and `render_report` is deliberately handed back
 * verbatim (see `agent.py`'s system prompt) — so the transcript has to render real
 * Markdown, tables included, not print `**1.03 %**` at the reader.
 *
 * Every element is mapped onto the charte rather than left to browser defaults: the
 * transcript reads as a document, so it obeys the same rules as a panel. Nothing here
 * introduces a colour — headings are ink, the one accent is the burnt rule on a
 * blockquote, which is the charte's own "out of envelope" mark.
 */
const COMPONENTS: Components = {
  p: ({ children }) => <p className="text-ctl leading-relaxed text-ink">{children}</p>,

  // Weight 500, never 600+: emphasis inside body text is the lightest step that reads.
  strong: ({ children }) => <strong className="font-medium text-ink">{children}</strong>,
  em: ({ children }) => <em className="not-italic text-ink2">{children}</em>,

  // Every heading collapses onto the two label sizes the charte has. A transcript is
  // not a page: it gets no third heading level and no size above 14px.
  h1: ({ children }) => <div className="cq-title mt-1">{children}</div>,
  h2: ({ children }) => <div className="cq-title mt-1">{children}</div>,
  h3: ({ children }) => <div className="cq-k mt-1">{children}</div>,
  h4: ({ children }) => <div className="cq-k mt-1">{children}</div>,

  ul: ({ children }) => <ul className="ml-4 list-disc space-y-1 text-ctl leading-relaxed marker:text-ink3">{children}</ul>,
  ol: ({ children }) => <ol className="ml-4 list-decimal space-y-1 text-ctl leading-relaxed marker:text-ink3">{children}</ol>,
  li: ({ children }) => <li className="text-ink">{children}</li>,

  // Numbers are mono wherever they appear — a tool name or a coefficient in the prose
  // is data, and data is set in the mono face.
  code: ({ children }) => (
    <code className="cq-num rounded-[4px] bg-paper px-1 py-px text-band text-ink">{children}</code>
  ),
  pre: ({ children }) => (
    <pre className="cq-num overflow-x-auto rounded-[8px] border border-line bg-paper px-3 py-2 text-band leading-relaxed">
      {children}
    </pre>
  ),

  // The charte's table: header cut off by an ink rule, rows by a --line hairline. No
  // zebra, no fill.
  table: ({ children }) => (
    <div className="-mx-0.5 overflow-x-auto">
      <table className="w-full border-collapse text-body">{children}</table>
    </div>
  ),
  th: ({ children }) => (
    <th className="cq-k border-b border-ink px-2 py-1.5 text-left align-bottom">{children}</th>
  ),
  td: ({ children }) => (
    <td className="cq-num border-b border-line px-2 py-1.5 text-band align-top">{children}</td>
  ),

  blockquote: ({ children }) => <blockquote className="cq-error text-ctl">{children}</blockquote>,
  hr: () => <hr className="border-line" />,
  a: ({ href, children }) => (
    <a href={href} className="text-ink underline decoration-line2 underline-offset-2 hover:decoration-or">
      {children}
    </a>
  ),
}

export function Markdown({ children }: { children: string }) {
  return (
    <div className="space-y-2.5">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={COMPONENTS}>
        {children}
      </ReactMarkdown>
    </div>
  )
}
