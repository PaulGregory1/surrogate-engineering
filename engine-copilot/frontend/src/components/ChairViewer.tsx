import { useEffect } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

/** One web mesh as exported by fcso-chairs/scripts/export_platform.py: positions quantised to
 *  uint16 in [-1, 1], indices as uint16/uint32, both base64. The first `n` faces are the
 *  largest connected component, the remaining `t - n` the stray ones. */
export type WebMesh = { p: string; i: string; i32: number; n: number; t: number }

// Same view as the report's PNG renders.
const CAM = { pos: [2.6, 1.6, -3.4] as const, target: [0, -0.05, 0] as const, half: 1.3 }

function b64(s: string) {
  const bin = atob(s)
  const out = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i)
  return out.buffer
}

function geometry(m: WebMesh) {
  const q = new Uint16Array(b64(m.p))
  const pos = new Float32Array(q.length)
  for (let i = 0; i < q.length; i++) pos[i] = (q[i] / 65535) * 2 - 1
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3))
  g.setIndex(new THREE.BufferAttribute(m.i32 ? new Uint32Array(b64(m.i)) : new Uint16Array(b64(m.i)), 1))
  g.addGroup(0, m.n * 3, 0)
  if (m.t > m.n) g.addGroup(m.n * 3, (m.t - m.n) * 3, 1)
  g.computeVertexNormals()
  return g
}

/**
 * A grid of meshes sharing one camera, drawn by a single WebGL canvas laid over the grid:
 * each cell element is a scissored viewport, so dragging anywhere turns every chair together.
 * `cells` maps a cell's `data-cell` key to its mesh; the caller owns the grid markup.
 */
export function useChairViewer(
  container: React.RefObject<HTMLDivElement | null>,
  canvas: React.RefObject<HTMLCanvasElement | null>,
  cells: Record<string, WebMesh> | null,
  /** bump to put the camera back to the default view */
  nonce = 0,
) {
  useEffect(() => {
    const root = container.current
    const cv = canvas.current
    if (!root || !cv || !cells) return

    const renderer = new THREE.WebGLRenderer({ canvas: cv, antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    renderer.setScissorTest(true)

    const camera = new THREE.OrthographicCamera(-CAM.half, CAM.half, CAM.half, -CAM.half, 0.01, 50)
    camera.position.set(...CAM.pos)
    camera.lookAt(...CAM.target)
    const controls = new OrbitControls(camera, root)
    // No wheel zoom: the grid sits in a scrolling page, and a wheel that zooms traps the scroll.
    controls.enableZoom = false
    controls.target.set(...CAM.target)
    controls.update()

    // Warm grey body; stray components in burnt — they are the defect being measured.
    const matMain = new THREE.MeshStandardMaterial({ color: 0xcfc8c1, roughness: 0.7, side: THREE.DoubleSide })
    const matStray = new THREE.MeshStandardMaterial({ color: 0xb03e00, roughness: 0.6, side: THREE.DoubleSide })
    const light = new THREE.DirectionalLight(0xffffff, 1.6)
    const hemi = new THREE.HemisphereLight(0xffffff, 0x6b6560, 1.1)

    const geoms: THREE.BufferGeometry[] = []
    const scenes = new Map<string, THREE.Scene>()
    for (const [key, m] of Object.entries(cells)) {
      const g = geometry(m)
      geoms.push(g)
      const s = new THREE.Scene()
      s.add(new THREE.Mesh(g, [matMain, matStray]))
      scenes.set(key, s)
    }

    let frame = 0
    const render = () => {
      frame = 0
      const box = root.getBoundingClientRect()
      const w = box.width
      const h = box.height
      renderer.setSize(w, h, false)
      renderer.setScissor(0, 0, w, h)
      renderer.setClearColor(0x000000, 0)
      renderer.clear()
      light.position.copy(camera.position).addScaledVector(camera.up, 1.2)
      for (const el of root.querySelectorAll<HTMLElement>('[data-cell]')) {
        const scene = scenes.get(el.dataset.cell!)
        if (!scene) continue
        const r = el.getBoundingClientRect()
        const x = r.left - box.left
        const y = box.bottom - r.bottom
        renderer.setViewport(x, y, r.width, r.height)
        renderer.setScissor(x, y, r.width, r.height)
        camera.left = (-CAM.half * r.width) / r.height
        camera.right = (CAM.half * r.width) / r.height
        camera.updateProjectionMatrix()
        scene.add(light, hemi)
        renderer.render(scene, camera)
      }
    }
    const request = () => {
      if (!frame) frame = requestAnimationFrame(render)
    }
    controls.addEventListener('change', request)
    const ro = new ResizeObserver(request)
    ro.observe(root)
    request()

    return () => {
      cancelAnimationFrame(frame)
      ro.disconnect()
      controls.dispose()
      geoms.forEach((g) => g.dispose())
      matMain.dispose()
      matStray.dispose()
      renderer.dispose()
    }
  }, [container, canvas, cells, nonce])
}
