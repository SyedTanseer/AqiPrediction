import { useState, useEffect, useCallback } from 'react'
import { connectWebSocket, fetchStatus, fetchModelInfo, toggleFan, getHealthAdvice } from '../utils/api'
import AQIGauge from '../components/AQIGauge'
import SensorPanel from '../components/SensorPanel'
import FanControl from '../components/FanControl'
import TrendChart from '../components/TrendChart'
import SystemStats from '../components/SystemStats'
import ModelInfo from '../components/ModelInfo'
import SubAQIBreakdown from '../components/SubAQIBreakdown'

export default function Dashboard() {
    const [status, setStatus] = useState(null)
    const [modelInfo, setModelInfo] = useState(null)
    const [connected, setConnected] = useState(false)

    const handleMessage = useCallback((msg) => {
        if (msg.type === 'init' || msg.type === 'update') {
            setStatus(msg.data)
            setConnected(true)
        }
    }, [])

    useEffect(() => {
        fetchStatus().then(setStatus).catch(() => setConnected(false))
        fetchModelInfo().then(setModelInfo).catch(() => { })
        const ws = connectWebSocket(handleMessage)
        return () => ws.close()
    }, [handleMessage])

    const handleFanToggle = async () => {
        if (!status) return
        const result = await toggleFan(!status.fan_on, status.fan_on ? 0 : 70)
        setStatus(prev => ({ ...prev, fan_on: result.fan_on, fan_speed: result.fan_speed }))
    }

    const aqi = status?.current_aqi || 0
    const category = status?.current_category || 'N/A'
    const color = status?.current_color || '#64748b'
    const advice = getHealthAdvice(aqi)
    const lastUpdate = status?.last_update
        ? new Date(status.last_update).toLocaleTimeString()
        : '--:--:--'

    return (
        <div className="dashboard-page">
            <header className="header">
                <div className="header-left">
                    <h1><span className="icon">🌫️</span> Air Quality Monitoring Dashboard</h1>
                    <p>Real-time air quality prediction and control system</p>
                </div>
                <div className="header-right">
                    <span>System Status:</span>
                    <span className={`status-badge ${connected && status?.system_online ? 'online' : 'offline'}`}>
                        <span className="status-dot"></span>
                        {connected && status?.system_online ? 'Online' : 'Offline'}
                    </span>
                    <span>Last updated: {lastUpdate}</span>
                </div>
            </header>

            <main className="main-content">
                {/* Row 1: AQI + Sensors + Fan */}
                <div className="cards-grid">
                    <div className="card aqi-card" id="aqi-card">
                        <div className="card-title"><span className="icon">📊</span> Current AQI</div>
                        <AQIGauge value={aqi} color={color} />
                        <div className="aqi-value" style={{ color }}>{aqi}</div>
                        <div className="aqi-category" style={{ color }}>{category}</div>
                        <div className="aqi-health-text" style={{ marginTop: '8px' }}>{advice.text}</div>
                    </div>

                    <div className="card" id="sensor-panel">
                        <div className="card-title"><span className="icon">📡</span> Sensor Status</div>
                        <SensorPanel readings={status?.sensor_readings} />
                    </div>

                    <div className="card fan-card" id="fan-control">
                        <div className="card-title"><span className="icon">🌀</span> Fan / Purifier Status</div>
                        <FanControl
                            fanOn={status?.fan_on || false}
                            fanSpeed={status?.fan_speed || 0}
                            onToggle={handleFanToggle}
                        />
                    </div>
                </div>

                {/* Row 2: Chart + Sub-AQI/Health/Model */}
                <div className="cards-grid-2">
                    <div className="card" id="trend-chart">
                        <div className="card-title"><span className="icon">📈</span> AQI Trend</div>
                        <TrendChart history={status?.prediction_history || []} />
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                        <div className="card" id="sub-aqi-breakdown">
                            <div className="card-title"><span className="icon">🧪</span> Pollutant Levels</div>
                            <SubAQIBreakdown
                                sensors={status?.sensor_readings}
                            />
                        </div>

                        <div className="card" id="health-advice">
                            <div className="card-title"><span className="icon">💡</span> Health Advisory</div>
                            <div className="health-advice" style={{ background: advice.bg, borderColor: advice.border }}>
                                <span className="health-icon">{advice.icon}</span>
                                <div className="health-text">
                                    <h4>{advice.title}</h4>
                                    <p>{advice.text}</p>
                                </div>
                            </div>
                        </div>

                        <div className="card" id="model-info">
                            <div className="card-title"><span className="icon">🧠</span> Model Info</div>
                            <ModelInfo info={modelInfo} />
                        </div>
                    </div>
                </div>

                {/* Row 3: System Stats */}
                <div className="card" id="system-stats">
                    <div className="card-title"><span className="icon">📉</span> System Statistics</div>
                    <SystemStats status={status} />
                </div>
            </main>
        </div>
    )
}
