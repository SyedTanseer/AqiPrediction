const SENSORS = [
    { key: 'PM25', label: 'PM2.5', unit: 'µg/m³', icon: '🟢' },
    { key: 'PM10', label: 'PM10', unit: 'µg/m³', icon: '🟢' },
    { key: 'NO2', label: 'NO₂', unit: 'µg/m³', icon: '🟢' },
    { key: 'SO2', label: 'SO₂', unit: 'µg/m³', icon: '🟢' },
    { key: 'CO', label: 'CO', unit: 'mg/m³', icon: '🟢' },
    { key: 'O3', label: 'O₃ (Ozone)', unit: 'µg/m³', icon: '🟢' },
]

export default function SensorPanel({ readings }) {
    return (
        <div className="sensor-list">
            {SENSORS.map(sensor => {
                const value = readings?.[sensor.key]
                const active = value !== undefined && value !== 0
                return (
                    <div className="sensor-item" key={sensor.key}>
                        <div className="sensor-left">
                            <div
                                className="sensor-dot"
                                style={{
                                    background: active ? '#22c55e' : '#64748b',
                                    boxShadow: active ? '0 0 8px rgba(34,197,94,0.5)' : 'none',
                                }}
                            />
                            <span className="sensor-name">{sensor.label}</span>
                        </div>
                        <div>
                            <span className="sensor-value">
                                {value !== undefined ? value : '--'}
                            </span>
                            <span className="sensor-unit">{sensor.unit}</span>
                        </div>
                    </div>
                )
            })}
        </div>
    )
}
