export default function SubAQIBreakdown({ sensors }) {
    if (!sensors || Object.keys(sensors).length === 0) {
        return <div style={{ color: '#64748b', fontSize: '0.85rem' }}>Waiting for sensor data...</div>
    }

    const pollutants = [
        { key: 'PM25', label: 'PM2.5', unit: 'µg/m³', max: 500 },
        { key: 'PM10', label: 'PM10', unit: 'µg/m³', max: 600 },
        { key: 'NO2', label: 'NO₂', unit: 'µg/m³', max: 400 },
        { key: 'SO2', label: 'SO₂', unit: 'µg/m³', max: 200 },
        { key: 'CO', label: 'CO', unit: 'mg/m³', max: 10 },
        { key: 'O3', label: 'O₃', unit: 'µg/m³', max: 200 },
    ]

    const getBarColor = (value, max) => {
        const ratio = value / max
        if (ratio <= 0.2) return '#22c55e'
        if (ratio <= 0.4) return '#eab308'
        if (ratio <= 0.6) return '#f97316'
        if (ratio <= 0.8) return '#ef4444'
        return '#a855f7'
    }

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {pollutants.map(({ key, label, unit, max }) => {
                const value = sensors[key] ?? 0
                return (
                    <div key={key}>
                        <div style={{
                            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                            marginBottom: '4px',
                        }}>
                            <span style={{
                                fontSize: '0.82rem', fontWeight: 400,
                                color: '#94a3b8',
                            }}>
                                {label}
                            </span>
                            <span style={{
                                fontSize: '0.82rem', fontWeight: 700,
                                color: getBarColor(value, max),
                            }}>
                                {value} <span style={{ fontWeight: 400, fontSize: '0.72rem' }}>{unit}</span>
                            </span>
                        </div>
                        <div style={{
                            width: '100%', height: '6px',
                            background: 'rgba(255,255,255,0.06)', borderRadius: '3px',
                            overflow: 'hidden',
                        }}>
                            <div style={{
                                width: `${Math.min(100, (value / max) * 100)}%`,
                                height: '100%', borderRadius: '3px',
                                background: getBarColor(value, max),
                                transition: 'width 0.5s ease, background 0.5s ease',
                            }} />
                        </div>
                    </div>
                )
            })}
        </div>
    )
}
