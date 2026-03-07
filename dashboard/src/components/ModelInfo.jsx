export default function ModelInfo({ info }) {
    if (!info) return <div style={{ color: '#64748b', fontSize: '0.85rem' }}>Loading model info...</div>

    const items = [
        { key: 'Architecture', val: 'Dense-only' },
        { key: 'Parameters', val: info.parameters?.toLocaleString() },
        { key: 'File Size', val: `${info.model_size_kb} KB` },
        { key: 'Target', val: 'CO (mg/m³)' },
        { key: 'Input Shape', val: '24 × 7' },
        { key: 'Status', val: info.loaded ? '✅ Loaded' : '❌ Error' },
    ]

    return (
        <div className="model-info-grid">
            {items.map((item, i) => (
                <div className="model-info-item" key={i}>
                    <div className="model-info-key">{item.key}</div>
                    <div className="model-info-val">{item.val}</div>
                </div>
            ))}
        </div>
    )
}
