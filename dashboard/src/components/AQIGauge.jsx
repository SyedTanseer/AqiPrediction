export default function AQIGauge({ value, color }) {
    const maxAqi = 500
    const clamped = Math.min(Math.max(value, 0), maxAqi)

    // 270° arc gauge (from 135° to 405°, i.e. gap at bottom)
    const startAngle = 135
    const endAngle = 405
    const totalArc = endAngle - startAngle // 270°
    const needleAngle = startAngle + (clamped / maxAqi) * totalArc

    const cx = 120, cy = 120, r = 95, innerR = 72

    const degToRad = (deg) => (Math.PI * deg) / 180

    // SVG arc path helper
    const arcPath = (from, to, radius) => {
        const fromRad = degToRad(from)
        const toRad = degToRad(to)
        const x1 = cx + radius * Math.cos(fromRad)
        const y1 = cy + radius * Math.sin(fromRad)
        const x2 = cx + radius * Math.cos(toRad)
        const y2 = cy + radius * Math.sin(toRad)
        const large = to - from > 180 ? 1 : 0
        return `M ${x1} ${y1} A ${radius} ${radius} 0 ${large} 1 ${x2} ${y2}`
    }

    // Color segments (6 AQI bands across 270°)
    const segments = [
        { endFrac: 50 / 500, color: '#22c55e' },   // Good
        { endFrac: 100 / 500, color: '#eab308' },  // Moderate
        { endFrac: 150 / 500, color: '#f97316' },  // USG
        { endFrac: 200 / 500, color: '#ef4444' },  // Unhealthy
        { endFrac: 300 / 500, color: '#a855f7' },  // Very Unhealthy
        { endFrac: 500 / 500, color: '#991b1b' },  // Hazardous
    ]

    let prevFrac = 0
    const arcs = segments.map((seg, i) => {
        const fromDeg = startAngle + prevFrac * totalArc
        const toDeg = startAngle + seg.endFrac * totalArc
        prevFrac = seg.endFrac
        return (
            <path
                key={i}
                d={arcPath(fromDeg, toDeg, r)}
                fill="none"
                stroke={seg.color}
                strokeWidth="14"
                strokeLinecap="butt"
                opacity="0.65"
            />
        )
    })

    // Tick marks
    const ticks = [0, 50, 100, 150, 200, 300, 500]
    const tickMarks = ticks.map((t, i) => {
        const angle = degToRad(startAngle + (t / maxAqi) * totalArc)
        const x1 = cx + (r + 4) * Math.cos(angle)
        const y1 = cy + (r + 4) * Math.sin(angle)
        const x2 = cx + (r - 6) * Math.cos(angle)
        const y2 = cy + (r - 6) * Math.sin(angle)
        const labelR = r + 16
        const lx = cx + labelR * Math.cos(angle)
        const ly = cy + labelR * Math.sin(angle)
        return (
            <g key={i}>
                <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="rgba(255,255,255,0.3)" strokeWidth="1.5" />
                <text x={lx} y={ly} textAnchor="middle" dominantBaseline="middle"
                    fill="rgba(255,255,255,0.35)" fontSize="8" fontWeight="500">{t}</text>
            </g>
        )
    })

    // Needle
    const needleRad = degToRad(needleAngle)
    const nx = cx + (r - 20) * Math.cos(needleRad)
    const ny = cy + (r - 20) * Math.sin(needleRad)
    // Needle tail (opposite direction, short)
    const tailX = cx - 12 * Math.cos(needleRad)
    const tailY = cy - 12 * Math.sin(needleRad)

    return (
        <div className="aqi-gauge" style={{ width: '240px', height: '240px', margin: '0 auto' }}>
            <svg viewBox="0 0 240 240" width="240" height="240">
                {/* Background arc */}
                <path
                    d={arcPath(startAngle, endAngle, r)}
                    fill="none"
                    stroke="rgba(255,255,255,0.06)"
                    strokeWidth="16"
                    strokeLinecap="round"
                />

                {/* Inner ring glow */}
                <circle cx={cx} cy={cy} r={innerR} fill="none"
                    stroke="rgba(255,255,255,0.03)" strokeWidth="1" />

                {/* Colored segments */}
                {arcs}

                {/* Tick marks + labels */}
                {tickMarks}

                {/* Needle shadow */}
                <line x1={tailX + 1} y1={tailY + 1} x2={nx + 1} y2={ny + 1}
                    stroke="rgba(0,0,0,0.3)" strokeWidth="3.5" strokeLinecap="round" />

                {/* Needle */}
                <line x1={tailX} y1={tailY} x2={nx} y2={ny}
                    stroke={color || '#fff'} strokeWidth="3" strokeLinecap="round"
                    style={{ transition: 'all 0.8s cubic-bezier(0.4, 0, 0.2, 1)' }}
                />

                {/* Center hub */}
                <circle cx={cx} cy={cy} r="8" fill={color || '#fff'} opacity="0.9" />
                <circle cx={cx} cy={cy} r="4" fill="#0a0e1a" />
            </svg>
        </div>
    )
}
