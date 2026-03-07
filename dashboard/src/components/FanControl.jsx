export default function FanControl({ fanOn, fanSpeed, onToggle }) {
    const speedLabel = fanSpeed === 0 ? 'OFF' : fanSpeed >= 90 ? 'MAX' : `${fanSpeed}%`

    const barColor = fanSpeed >= 80
        ? 'linear-gradient(90deg, #22c55e, #ef4444)'
        : fanSpeed >= 50
            ? 'linear-gradient(90deg, #22c55e, #eab308)'
            : 'linear-gradient(90deg, #22c55e, #22d3ee)'

    return (
        <>
            <div className="fan-icon-wrapper">
                <span
                    className={`fan-icon ${fanOn ? 'spinning' : ''}`}
                    style={{ animationDuration: fanOn ? `${Math.max(0.2, 1.5 - fanSpeed / 100)}s` : 'unset' }}
                >
                    🌀
                </span>
            </div>
            <div className="fan-speed-label">{speedLabel}</div>
            <div className="fan-speed-sublabel">Fan Speed</div>
            <div className="fan-speed-bar">
                <div
                    className="fan-speed-fill"
                    style={{
                        width: `${fanSpeed}%`,
                        background: barColor,
                    }}
                />
            </div>
            <button
                className={`fan-toggle-btn ${fanOn ? 'active' : ''}`}
                onClick={onToggle}
                id="fan-toggle-btn"
            >
                ⏻ Toggle Fan
            </button>
        </>
    )
}
