import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'

const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null
    const d = payload[0].payload
    return (
        <div style={{
            background: 'rgba(15, 23, 42, 0.95)',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: '8px',
            padding: '10px 14px',
            fontSize: '0.8rem',
        }}>
            <div style={{ color: '#94a3b8', marginBottom: 4 }}>
                {new Date(d.timestamp).toLocaleTimeString()}
            </div>
            <div style={{ color: d.color, fontWeight: 700, fontSize: '1rem' }}>
                AQI: {d.aqi}
            </div>
            <div style={{ color: '#94a3b8', fontSize: '0.75rem' }}>{d.category}</div>
            <div style={{ color: '#22d3ee', marginTop: 4 }}>
                CO: {d.co_mg_m3} mg/m³
            </div>
        </div>
    )
}

export default function TrendChart({ history }) {
    const data = history.map((h, i) => ({
        ...h,
        index: i,
        time: new Date(h.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
    }))

    return (
        <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data} margin={{ top: 5, right: 10, left: -15, bottom: 5 }}>
                    <defs>
                        <linearGradient id="aqiGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#818cf8" stopOpacity={0.4} />
                            <stop offset="95%" stopColor="#818cf8" stopOpacity={0} />
                        </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis
                        dataKey="time"
                        stroke="#64748b"
                        tick={{ fontSize: 11 }}
                        interval="preserveStartEnd"
                    />
                    <YAxis
                        stroke="#64748b"
                        tick={{ fontSize: 11 }}
                        domain={[0, 'auto']}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <ReferenceLine y={50} stroke="#22c55e" strokeDasharray="5 5" strokeOpacity={0.4} />
                    <ReferenceLine y={100} stroke="#eab308" strokeDasharray="5 5" strokeOpacity={0.4} />
                    <ReferenceLine y={200} stroke="#ef4444" strokeDasharray="5 5" strokeOpacity={0.4} />
                    <Area
                        type="monotone"
                        dataKey="aqi"
                        stroke="#818cf8"
                        strokeWidth={2}
                        fill="url(#aqiGradient)"
                        animationDuration={300}
                    />
                </AreaChart>
            </ResponsiveContainer>
        </div>
    )
}
