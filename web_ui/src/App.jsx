import { useState, useEffect, useRef, useCallback } from 'react'

const API_BASE = 'http://localhost:8000'

const CLASS_INFO = [
  { idx: 0, name: 'class_1_fine_texture', display: 'Fine Texture', color: '#3AB4F2', desc: 'Dense, fine weave patterns' },
  { idx: 1, name: 'class_2_stochastic_texture', display: 'Stochastic', color: '#F2DC3A', desc: 'Random, irregular patterns' },
  { idx: 2, name: 'class_3_periodic_texture', display: 'Periodic', color: '#3AF252', desc: 'Repeating geometric patterns' },
  { idx: 3, name: 'class_4_printed_nonperiodic', display: 'Printed', color: '#C864FF', desc: 'Complex, non-repeating designs' },
]

function App() {
  const [status, setStatus] = useState({
    class_idx: 0,
    class_display: 'Fine Texture',
    class_color: '#3AB4F2',
    paused: false,
    frame_count: 0,
    defect_count: 0,
    good_count: 0,
    fps: 0,
    latency_ms: 0,
    defect_prob: 0,
    is_defect: false,
    defect_rate: 0,
    threshold: 0.5,
  })
  const [connected, setConnected] = useState(false)
  const [switching, setSwitching] = useState(false)
  const imgRef = useRef(null)

  // Poll status from backend
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/status`)
        if (res.ok) {
          const data = await res.json()
          setStatus(data)
          setConnected(true)
        }
      } catch {
        setConnected(false)
      }
    }, 500)
    return () => clearInterval(interval)
  }, [])

  // API calls
  const selectClass = useCallback(async (idx) => {
    setSwitching(true)
    try {
      await fetch(`${API_BASE}/api/select-class`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ class_idx: idx }),
      })
    } catch (e) { console.error(e) }
    setSwitching(false)
  }, [])

  const togglePause = useCallback(async () => {
    try { await fetch(`${API_BASE}/api/pause`, { method: 'POST' }) }
    catch (e) { console.error(e) }
  }, [])

  const triggerHeatmap = useCallback(async () => {
    try { await fetch(`${API_BASE}/api/heatmap`, { method: 'POST' }) }
    catch (e) { console.error(e) }
  }, [])

  const resetStats = useCallback(async () => {
    try { await fetch(`${API_BASE}/api/reset-stats`, { method: 'POST' }) }
    catch (e) { console.error(e) }
  }, [])

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e) => {
      if (e.key === 'p' || e.key === 'P') togglePause()
      else if (e.key === 'h' || e.key === 'H') triggerHeatmap()
      else if (e.key === 'r' || e.key === 'R') resetStats()
      else if (e.key >= '1' && e.key <= '4') selectClass(parseInt(e.key) - 1)
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [togglePause, triggerHeatmap, resetStats, selectClass])

  // Gauge calculation
  const gaugeRadius = 58
  const gaugeCircumference = 2 * Math.PI * gaugeRadius
  const gaugeOffset = gaugeCircumference * (1 - status.defect_prob)
  const gaugeColor = status.defect_prob >= status.threshold ? '#ef4444' : '#22c55e'

  const rateBarColor = status.defect_rate > 50 ? '#ef4444' : status.defect_rate > 25 ? '#f59e0b' : '#22c55e'

  return (
    <div className="app">
      {/* ═══════ HEADER ═══════ */}
      <header className="header">
        <div className="header-left">
          <div className="header-logo">TD</div>
          <span className="header-title">Textile Defect Detection</span>
        </div>
        <div className="header-right">
          <div className="header-metric">
            <span className="header-metric-label">FPS</span>
            <span className="header-metric-value">{status.fps}</span>
          </div>
          <div className="header-metric">
            <span className="header-metric-label">Latency</span>
            <span className="header-metric-value">{status.latency_ms}ms</span>
          </div>
          <div className={`live-badge ${status.paused ? 'paused' : 'active'}`}>
            <span className="live-dot"></span>
            {status.paused ? 'Paused' : 'Live'}
          </div>
        </div>
      </header>

      {/* ═══════ MAIN CONTENT ═══════ */}
      <main className="main-content">

        {/* ─── LEFT: Class Selector ─── */}
        <div className="left-panel">
          <div className="glass-card">
            <div className="card-header">
              <span className="card-header-icon">🧵</span>
              <span className="card-header-title">Fabric Class</span>
            </div>
            <div className="card-body">
              <div className="class-cards">
                {CLASS_INFO.map((cls) => (
                  <div
                    key={cls.idx}
                    className={`class-card ${status.class_idx === cls.idx ? 'selected' : ''}`}
                    style={{ '--card-accent-color': cls.color }}
                    onClick={() => selectClass(cls.idx)}
                  >
                    <div className="class-card-name">
                      <span className="class-card-indicator" style={{ background: cls.color }}></span>
                      {cls.display}
                    </div>
                    <div className="class-card-desc">{cls.desc}</div>
                  </div>
                ))}
              </div>
              {switching && <p style={{ fontSize: 11, color: '#f59e0b', marginTop: 10, textAlign: 'center' }}>Loading model...</p>}
            </div>
          </div>

          {/* Threshold info */}
          <div className="glass-card">
            <div className="card-header">
              <span className="card-header-icon">⚙️</span>
              <span className="card-header-title">Model Info</span>
            </div>
            <div className="card-body">
              <div className="stat-row">
                <span className="stat-label">Active Model</span>
                <span className="stat-value blue">{CLASS_INFO[status.class_idx]?.display}</span>
              </div>
              <div className="stat-row">
                <span className="stat-label">Threshold</span>
                <span className="stat-value" style={{ color: '#06b6d4' }}>{(status.threshold * 100).toFixed(0)}%</span>
              </div>
            </div>
          </div>
        </div>

        {/* ─── CENTER: Video Feed ─── */}
        <div className="video-container">
          {connected ? (
            <>
              <img
                ref={imgRef}
                className="video-feed"
                src={`${API_BASE}/api/video-feed`}
                alt="Live video feed"
              />
              <div className={`video-overlay-status ${status.is_defect ? 'defect' : 'good'}`}>
                {status.is_defect ? '⚠ DEFECT' : '✓ GOOD'}
              </div>
            </>
          ) : (
            <div className="no-connection">
              <div className="spinner"></div>
              <p>Connecting to backend server...</p>
              <p style={{ fontSize: 12, color: '#475569' }}>
                Run: python backend_server.py
              </p>
            </div>
          )}
        </div>

        {/* ─── RIGHT: Stats Dashboard ─── */}
        <div className="right-panel">
          {/* Defect Probability Gauge */}
          <div className="glass-card">
            <div className="card-header">
              <span className="card-header-icon">📊</span>
              <span className="card-header-title">Defect Probability</span>
            </div>
            <div className="card-body">
              <div className="gauge-container">
                <div className="gauge-ring">
                  <svg width="140" height="140" viewBox="0 0 140 140">
                    <circle className="gauge-bg" cx="70" cy="70" r={gaugeRadius} />
                    <circle
                      className="gauge-fill"
                      cx="70" cy="70" r={gaugeRadius}
                      stroke={gaugeColor}
                      strokeDasharray={gaugeCircumference}
                      strokeDashoffset={gaugeOffset}
                    />
                  </svg>
                  <div className="gauge-center">
                    <div className="gauge-value" style={{ color: gaugeColor }}>
                      {(status.defect_prob * 100).toFixed(1)}%
                    </div>
                    <div className="gauge-label">Probability</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Session Stats */}
          <div className="glass-card">
            <div className="card-header">
              <span className="card-header-icon">📈</span>
              <span className="card-header-title">Session Stats</span>
            </div>
            <div className="card-body">
              <div className="stat-row">
                <span className="stat-label">Frames Inspected</span>
                <span className="stat-value">{status.frame_count.toLocaleString()}</span>
              </div>
              <div className="stat-row">
                <span className="stat-label">Defects Found</span>
                <span className="stat-value red">{status.defect_count.toLocaleString()}</span>
              </div>
              <div className="stat-row">
                <span className="stat-label">Good Frames</span>
                <span className="stat-value green">{status.good_count.toLocaleString()}</span>
              </div>
              <div className="stat-row">
                <span className="stat-label">Defect Rate</span>
                <span className="stat-value amber">{status.defect_rate}%</span>
              </div>
              <div className="rate-bar">
                <div className="rate-bar-fill" style={{
                  width: `${Math.min(status.defect_rate, 100)}%`,
                  background: rateBarColor,
                }}></div>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* ═══════ BOTTOM CONTROLS ═══════ */}
      <footer className="controls-bar">
        <button className="control-btn primary" onClick={togglePause}>
          <span className="icon">{status.paused ? '▶' : '⏸'}</span>
          {status.paused ? 'Resume' : 'Pause'}
          <span className="kbd">P</span>
        </button>
        <button className="control-btn" onClick={triggerHeatmap}>
          <span className="icon">🔥</span>
          Heatmap
          <span className="kbd">H</span>
        </button>
        <button className="control-btn danger" onClick={resetStats}>
          <span className="icon">↺</span>
          Reset Stats
          <span className="kbd">R</span>
        </button>
      </footer>
    </div>
  )
}

export default App
