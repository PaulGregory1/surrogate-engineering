import { motion } from 'framer-motion'
import { COEFF_RANGE, boundaryXY, r0ForArea } from '../geometry'

const LABELS = ['a2', 'b2', 'a3', 'b3', 'a4', 'b4']

export function ShapeEditor({
  coeffs,
  onChange,
  reynolds,
  onReynoldsChange,
  targetArea,
  readOnly = false,
  label,
}: {
  coeffs: number[]
  onChange: (coeffs: number[]) => void
  reynolds: number
  onReynoldsChange: (re: number) => void
  targetArea: number
  readOnly?: boolean
  label?: string
}) {
  const r0 = r0ForArea(coeffs, targetArea)
  const { x, y } = boundaryXY(r0, coeffs)
  const path = x.map((xi, i) => `${i === 0 ? 'M' : 'L'} ${xi.toFixed(2)} ${y[i].toFixed(2)}`).join(' ') + ' Z'
  const extent = r0 * 1.6

  function setCoeff(i: number, v: number) {
    const next = [...coeffs]
    next[i] = v
    onChange(next)
  }

  return (
    <div className="nc-card p-4 flex flex-col gap-4">
      {label && <div className="text-sm font-medium" style={{ color: 'var(--text-muted)' }}>{label}</div>}
      <svg viewBox={`${-extent} ${-extent} ${extent * 2} ${extent * 2}`} className="w-full aspect-square rounded-md" style={{ background: 'var(--bg)' }}>
        <line x1={-extent} y1="0" x2={extent} y2="0" stroke="var(--border)" strokeWidth={extent / 200} />
        <line x1="0" y1={-extent} x2="0" y2={extent} stroke="var(--border)" strokeWidth={extent / 200} />
        <motion.path
          d={path}
          fill="color-mix(in srgb, var(--accent) 18%, transparent)"
          stroke="var(--accent)"
          strokeWidth={extent / 90}
          animate={{ d: path }}
          transition={{ type: 'spring', stiffness: 220, damping: 26 }}
        />
      </svg>

      {!readOnly && (
        <div className="grid grid-cols-2 gap-x-4 gap-y-2">
          {LABELS.map((lab, i) => (
            <label key={lab} className="flex items-center gap-2 text-xs" style={{ color: 'var(--text-muted)' }}>
              <span className="w-6 font-mono">{lab}</span>
              <input
                type="range"
                min={-COEFF_RANGE}
                max={COEFF_RANGE}
                step={0.005}
                value={coeffs[i]}
                onChange={(e) => setCoeff(i, parseFloat(e.target.value))}
                className="flex-1 accent-cyan-400"
              />
            </label>
          ))}
        </div>
      )}

      {!readOnly && (
        <label className="flex items-center gap-2 text-xs" style={{ color: 'var(--text-muted)' }}>
          <span className="w-14">Re = {reynolds.toFixed(0)}</span>
          <input
            type="range"
            min={10}
            max={200}
            step={1}
            value={reynolds}
            onChange={(e) => onReynoldsChange(parseFloat(e.target.value))}
            className="flex-1 accent-amber-400"
          />
        </label>
      )}

      {!readOnly && (
        <div className="flex gap-2">
          <button
            onClick={() => onChange([0, 0, 0, 0, 0, 0])}
            className="nc-card px-3 py-1.5 text-xs hover:brightness-125 transition"
          >
            Reset
          </button>
          <button
            onClick={() => onChange(coeffs.map(() => (Math.random() * 2 - 1) * COEFF_RANGE * 0.7))}
            className="nc-card px-3 py-1.5 text-xs hover:brightness-125 transition"
          >
            Randomize
          </button>
        </div>
      )}
    </div>
  )
}
