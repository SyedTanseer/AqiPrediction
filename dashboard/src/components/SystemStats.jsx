export default function SystemStats({ status }) {
    const stats = [
        {
            label: 'Predictions Today',
            value: status?.predictions_count ?? '--',
            color: '#818cf8',
        },
        {
            label: 'Dominant Pollutant',
            value: status?.dominant_pollutant ?? '--',
            color: '#fb7185',
        },
        {
            label: 'System Uptime (hrs)',
            value: status?.uptime_hours ?? '--',
            color: '#34d399',
        },
        {
            label: 'Predicted CO (mg/m³)',
            value: status?.predicted_co_mg_m3 !== undefined
                ? status.predicted_co_mg_m3.toFixed(2)
                : '--',
            color: '#fbbf24',
        },
    ]

    return (
        <div className="stats-grid">
            {stats.map((stat, i) => (
                <div className="stat-item" key={i}>
                    <div className="stat-value" style={{ color: stat.color }}>{stat.value}</div>
                    <div className="stat-label">{stat.label}</div>
                </div>
            ))}
        </div>
    )
}
