import { useState, useEffect, useRef } from 'react'
import { sendSensorData, resetSystem, connectWebSocket } from '../utils/api'

const SENSOR_CONFIG = [
    { key: 'pm25', label: 'PM2.5 (Fine Particles)', unit: 'µg/m³', min: 0, max: 500, step: 1, default: 50 },
    { key: 'pm10', label: 'PM10 (Coarse Particles)', unit: 'µg/m³', min: 0, max: 600, step: 1, default: 80 },
    { key: 'no2', label: 'NO₂ (Nitrogen Dioxide)', unit: 'µg/m³', min: 0, max: 400, step: 1, default: 30 },
    { key: 'so2', label: 'SO₂ (Sulfur Dioxide)', unit: 'µg/m³', min: 0, max: 200, step: 1, default: 15 },
    { key: 'co', label: 'CO (Carbon Monoxide)', unit: 'mg/m³', min: 0, max: 10, step: 0.1, default: 1.0 },
    { key: 'o3', label: 'O₃ (Ozone)', unit: 'µg/m³', min: 0, max: 200, step: 1, default: 40 },
]

// generate realistic Indian city diurnal patterns
function generateRealisticValue(sensor, hour, noise) {
    const diurnal = Math.sin((hour / 24) * Math.PI * 2 - Math.PI / 2)
    switch (sensor) {
        case 'pm25':
            return Math.max(5, 55 + diurnal * 35 + noise * 20)
        case 'pm10':
            return Math.max(10, 90 + diurnal * 50 + noise * 25)
        case 'no2':
            return Math.max(2, 35 + diurnal * 20 + noise * 10)
        case 'so2':
            return Math.max(1, 18 + diurnal * 8 + noise * 5)
        case 'co':
            return Math.max(0.1, 1.2 + diurnal * 0.6 + noise * 0.3)
        case 'o3':
            return Math.max(2, 45 + diurnal * 25 + noise * 12)
        default:
            return 0
    }
}

export default function Simulator() {
    const [values, setValues] = useState(
        Object.fromEntries(SENSOR_CONFIG.map(s => [s.key, s.default]))
    )
    const [autoMode, setAutoMode] = useState(false)
    const [lastPrediction, setLastPrediction] = useState(null)
    const [sendCount, setSendCount] = useState(0)
    const [status, setStatus] = useState(null)
    const [speed, setSpeed] = useState(1000)
    const autoRef = useRef(false)
    const hourRef = useRef(0)

    // WebSocket for receiving status updates
    useEffect(() => {
        const ws = connectWebSocket((msg) => {
            if (msg.type === 'init' || msg.type === 'update') {
                setStatus(msg.data)
            }
        })
        return () => ws.close()
    }, [])

    // Auto simulation loop
    useEffect(() => {
        autoRef.current = autoMode
        if (!autoMode) return

        const interval = setInterval(async () => {
            if (!autoRef.current) return

            hourRef.current = (hourRef.current + 0.5) % 24
            const noise = (Math.random() - 0.5) * 2

            const newValues = {}
            SENSOR_CONFIG.forEach(s => {
                newValues[s.key] = parseFloat(generateRealisticValue(s.key, hourRef.current, noise).toFixed(2))
            })
            setValues(newValues)

            try {
                const result = await sendSensorData(
                    newValues.pm25, newValues.pm10, newValues.no2,
                    newValues.so2, newValues.co, newValues.o3
                )
                setSendCount(c => c + 1)
                if (result.prediction) {
                    setLastPrediction(result.prediction)
                }
            } catch (e) {
                console.error('Send failed:', e)
            }
        }, speed)

        return () => clearInterval(interval)
    }, [autoMode, speed])

    const handleSliderChange = (key, val) => {
        setValues(prev => ({ ...prev, [key]: parseFloat(val) }))
    }

    const handleManualSend = async () => {
        try {
            const result = await sendSensorData(
                values.pm25, values.pm10, values.no2,
                values.so2, values.co, values.o3
            )
            setSendCount(c => c + 1)
            if (result.prediction) {
                setLastPrediction(result.prediction)
            }
        } catch (e) {
            console.error('Send failed:', e)
        }
    }

    const handleReset = async () => {
        await resetSystem()
        setLastPrediction(null)
        setSendCount(0)
        setAutoMode(false)
    }

    return (
        <div className="simulator-page">
            <header className="header">
                <div className="header-left">
                    <h1><span className="icon">🎛️</span> AQI Sensor Control Panel</h1>
                    <p>Simulate sensor readings and feed them into the AQI prediction model</p>
                </div>
                <div className="header-right">
                    <a href="/" target="_blank" rel="noopener noreferrer"
                        style={{ color: '#818cf8', textDecoration: 'none', fontSize: '0.85rem' }}>
                        📊 Open Dashboard →
                    </a>
                    <span className={`auto-badge ${autoMode ? 'running' : 'stopped'}`}>
                        <span className="status-dot"></span>
                        {autoMode ? 'Auto Running' : 'Manual Mode'}
                    </span>
                </div>
            </header>

            <main className="main-content">
                <div className="sim-layout">
                    {/* Left: Sensor Sliders */}
                    <div className="card">
                        <div className="card-title"><span className="icon">📡</span> Sensor Values</div>
                        {SENSOR_CONFIG.map(sensor => (
                            <div className="slider-group" key={sensor.key}>
                                <div className="slider-header">
                                    <span className="slider-label">{sensor.label}</span>
                                    <span className="slider-value-display">
                                        {values[sensor.key]} <span style={{ color: '#64748b', fontSize: '0.75rem' }}>{sensor.unit}</span>
                                    </span>
                                </div>
                                <input
                                    type="range"
                                    min={sensor.min}
                                    max={sensor.max}
                                    step={sensor.step}
                                    value={values[sensor.key]}
                                    onChange={e => handleSliderChange(sensor.key, e.target.value)}
                                    disabled={autoMode}
                                    id={`slider-${sensor.key}`}
                                />
                            </div>
                        ))}

                        <div className="slider-group">
                            <div className="slider-header">
                                <span className="slider-label">Auto Speed</span>
                                <span className="slider-value-display">{speed}ms</span>
                            </div>
                            <input
                                type="range" min={200} max={3000} step={100} value={speed}
                                onChange={e => setSpeed(parseInt(e.target.value))}
                                id="slider-speed"
                            />
                        </div>

                        <div className="sim-controls-row">
                            <button className="btn btn-primary" onClick={handleManualSend} disabled={autoMode}
                                id="btn-send">
                                📤 Send Reading
                            </button>
                            <button
                                className={`btn ${autoMode ? 'btn-danger' : 'btn-success'}`}
                                onClick={() => setAutoMode(!autoMode)}
                                id="btn-auto"
                            >
                                {autoMode ? '⏹ Stop Auto' : '▶ Start Auto'}
                            </button>
                            <button className="btn" onClick={handleReset} id="btn-reset">🔄 Reset</button>
                        </div>
                    </div>

                    {/* Right: Status & Prediction */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                        {/* Prediction Result */}
                        <div className="card">
                            <div className="card-title"><span className="icon">🎯</span> AQI Prediction</div>
                            {lastPrediction ? (
                                <div style={{ textAlign: 'center' }}>
                                    <div className="aqi-value" style={{ color: lastPrediction.color, fontSize: '3rem' }}>
                                        {lastPrediction.aqi}
                                    </div>
                                    <div className="aqi-category" style={{ color: lastPrediction.color }}>
                                        {lastPrediction.category}
                                    </div>
                                    <div style={{ color: '#94a3b8', fontSize: '0.8rem', marginTop: 8 }}>
                                        Direct AQI prediction from 6 pollutant sensors
                                    </div>
                                </div>
                            ) : (
                                <div style={{ textAlign: 'center', color: '#64748b', padding: '30px 0' }}>
                                    <div style={{ fontSize: '2rem', marginBottom: 8 }}>🔮</div>
                                    <div>Send a reading to get a prediction</div>
                                    <div style={{ fontSize: '0.8rem', marginTop: 4 }}>
                                        Instant AQI prediction — no buffer needed
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Transmission Stats */}
                        <div className="card">
                            <div className="card-title"><span className="icon">📊</span> Session Stats</div>
                            <div className="stats-grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
                                <div className="stat-item">
                                    <div className="stat-value" style={{ fontSize: '1.5rem', color: '#818cf8' }}>
                                        {sendCount}
                                    </div>
                                    <div className="stat-label">Readings Sent</div>
                                </div>
                                <div className="stat-item">
                                    <div className="stat-value" style={{ fontSize: '1.5rem', color: '#22d3ee' }}>
                                        {status?.predictions_count ?? 0}
                                    </div>
                                    <div className="stat-label">Predictions</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    )
}
