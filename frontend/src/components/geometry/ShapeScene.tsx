/**
 * The Shape view's 3D scene: every example and the class means on the class means' top three
 * principal components, means joined in value order (a closed loop for cyclic concepts). The
 * Atlas view draws a community's tokens with it (its own colors, titles, and no path).
 *
 * Loaded lazily (React.lazy) so other pages don't pay for three.js. Renders on demand: a frame
 * only when the data, the camera or the hover changes.
 */

import { useLayoutEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { Canvas, useThree, type ThreeEvent } from '@react-three/fiber'
import { Html, Line, OrbitControls } from '@react-three/drei'

export interface ShapeSceneProps {
  /** [n, 3] every example */
  points: number[][]
  /** [k, 3] the class means */
  means: number[][]
  /** A color per value; example i has value i % k */
  colors: string[]
  labels: string[]
  closed: boolean
  showPoints: boolean
  /** Explained variance ratio of PC1–3 */
  explained: number[]
  /** Text for example i (the template with its value filled in) */
  exampleText: (i: number) => string
  /** Point i's color and title (default: its value's, example i having value i % k) */
  pointColor?: (i: number) => string
  pointTitle?: (i: number) => string
  /** Join the means in order (default true) */
  connect?: boolean
  ink: string
  muted: string
  surface: string
  height?: number
}

// Values labeled at most (every nth value beyond this)
const MAX_LABELS = 24
const POINT_RADIUS = 0.025
const MEAN_RADIUS = 0.07

function Points({ points, colorOf, onHover }: { points: number[][]; colorOf: (i: number) => string; onHover: (i: number | null) => void }) {
  const mesh = useRef<THREE.InstancedMesh>(null)
  const invalidate = useThree((state) => state.invalidate)
  // Instance colors exist from the first render, so the material compiles with them
  const colorArray = useMemo(() => {
    const array = new Float32Array(points.length * 3)
    const color = new THREE.Color()
    for (let i = 0; i < points.length; i++) {
      color.set(colorOf(i))
      array.set([color.r, color.g, color.b], i * 3)
    }
    return array
  }, [points.length, colorOf])

  useLayoutEffect(() => {
    const instanced = mesh.current
    if (!instanced) return
    const dummy = new THREE.Object3D()
    points.forEach(([x, y, z], i) => {
      dummy.position.set(x, y, z ?? 0)
      dummy.updateMatrix()
      instanced.setMatrixAt(i, dummy.matrix)
    })
    instanced.instanceMatrix.needsUpdate = true
    // Hover picking tests this sphere first
    instanced.computeBoundingSphere()
    invalidate()
  }, [points, invalidate])

  return (
    <instancedMesh
      key={points.length}
      ref={mesh}
      args={[undefined, undefined, points.length]}
      onPointerMove={(e: ThreeEvent<PointerEvent>) => {
        e.stopPropagation()
        onHover(e.instanceId ?? null)
      }}
      onPointerOut={() => onHover(null)}
    >
      <sphereGeometry args={[POINT_RADIUS, 8, 8]} />
      <meshStandardMaterial transparent opacity={0.55} />
      <instancedBufferAttribute attach="instanceColor" args={[colorArray, 3]} />
    </instancedMesh>
  )
}

function Label({ position, text, color, strong }: { position: [number, number, number]; text: string; color: string; strong?: boolean }) {
  return (
    <Html position={position} center style={{ pointerEvents: 'none' }} zIndexRange={[10, 0]}>
      <span style={{ color, fontSize: 11, fontWeight: strong ? 600 : 500, whiteSpace: 'nowrap', textShadow: 'none' }}>{text}</span>
    </Html>
  )
}

export default function ShapeScene({
  points,
  means,
  colors,
  labels,
  closed,
  showPoints,
  explained,
  exampleText,
  pointColor,
  pointTitle,
  connect = true,
  ink,
  muted,
  surface,
  height = 380,
}: ShapeSceneProps) {
  const [hoverPoint, setHoverPoint] = useState<number | null>(null)
  const [hoverMean, setHoverMean] = useState<number | null>(null)
  const k = means.length
  const colorOf = useMemo(() => pointColor ?? ((i: number) => colors[i % k]), [pointColor, colors, k])
  const titleOf = pointTitle ?? ((i: number) => labels[i % k])
  const path = useMemo(() => {
    if (!connect) return []
    const line = means.map(([x, y, z]) => [x, y, z ?? 0] as [number, number, number])
    return closed && line.length > 2 ? [...line, line[0]] : line
  }, [means, closed, connect])
  const labelEvery = Math.max(1, Math.ceil(k / MAX_LABELS))
  // Class means have RMS radius 1 over all their components, about 0.6–0.9 on the top three
  const axis = 1.25
  const axisLabel = (i: number) => `PC${i + 1} ${Math.round((explained[i] ?? 0) * 100)}%`

  return (
    <div style={{ height, position: 'relative', cursor: hoverPoint != null || hoverMean != null ? 'pointer' : 'grab' }}>
      <Canvas frameloop="demand" dpr={[1, 2]} camera={{ position: [1.55, 1.15, 1.95], fov: 45, near: 0.01, far: 100 }}>
        <ambientLight intensity={0.85} />
        <directionalLight position={[3, 5, 4]} intensity={0.7} />

        {[0, 1, 2].map((i) => {
          const end: [number, number, number] = [0, 0, 0]
          end[i] = axis
          return (
            <group key={i}>
              <Line points={[[0, 0, 0], end]} color={muted} lineWidth={1} transparent opacity={0.6} />
              <Label position={end} text={axisLabel(i)} color={muted} />
            </group>
          )
        })}

        {showPoints && <Points points={points} colorOf={colorOf} onHover={setHoverPoint} />}

        {path.length > 1 && <Line points={path} color={ink} lineWidth={1.5} transparent opacity={0.7} />}
        {means.map(([x, y, z], v) => (
          <mesh
            key={v}
            position={[x, y, z ?? 0]}
            onPointerMove={(e) => {
              e.stopPropagation()
              setHoverMean(v)
            }}
            onPointerOut={() => setHoverMean(null)}
          >
            <sphereGeometry args={[hoverMean === v ? MEAN_RADIUS * 1.4 : MEAN_RADIUS, 20, 20]} />
            <meshStandardMaterial color={colors[v]} emissive={colors[v]} emissiveIntensity={0.25} />
          </mesh>
        ))}
        {means.map(([x, y, z], v) =>
          v % labelEvery === 0 || v === k - 1 ? (
            <Label key={v} position={[x * 1.12 + 0.02, y * 1.12 + 0.09, (z ?? 0) * 1.12]} text={labels[v]} color={ink} strong />
          ) : null,
        )}

        {hoverPoint != null && points[hoverPoint] && (
          <Html position={points[hoverPoint] as [number, number, number]} style={{ pointerEvents: 'none' }}>
            <div className="gl-scene-tip" style={{ background: surface, color: ink, borderColor: muted }}>
              <strong>{titleOf(hoverPoint)}</strong>
              <div>{exampleText(hoverPoint)}</div>
            </div>
          </Html>
        )}
        {hoverMean != null && (
          <Html position={means[hoverMean] as [number, number, number]} style={{ pointerEvents: 'none' }}>
            <div className="gl-scene-tip" style={{ background: surface, color: ink, borderColor: muted }}>
              <strong>{labels[hoverMean]}</strong>
              <div>Class mean</div>
            </div>
          </Html>
        )}

        <OrbitControls makeDefault enableDamping dampingFactor={0.12} minDistance={0.8} maxDistance={20} />
      </Canvas>
      <style>{`
        .gl-scene-tip {
          transform: translate(10px, -50%);
          padding: 0.35rem 0.5rem;
          border: 1px solid;
          font-size: 0.72rem;
          line-height: 1.35;
          width: max-content;
          max-width: 240px;
          box-shadow: 0 6px 24px rgba(0, 0, 0, 0.18);
        }
      `}</style>
    </div>
  )
}
