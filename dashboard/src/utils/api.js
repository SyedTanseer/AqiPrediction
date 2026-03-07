const API_BASE = 'http://localhost:8000'
const WS_URL = 'ws://localhost:8000/ws'

export async function fetchStatus() {
    const res = await fetch(`${API_BASE}/api/status`)
    return res.json()
}

export async function fetchModelInfo() {
    const res = await fetch(`${API_BASE}/api/model-info`)
    return res.json()
}

export async function sendSensorData(pm25, pm10, no2, so2, co, o3) {
    const res = await fetch(`${API_BASE}/api/sensor-data`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pm25, pm10, no2, so2, co, o3 }),
    })
    return res.json()
}

export async function toggleFan(on, speed) {
    const res = await fetch(`${API_BASE}/api/fan/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ on, speed }),
    })
    return res.json()
}

export async function resetSystem() {
    const res = await fetch(`${API_BASE}/api/reset`, { method: 'POST' })
    return res.json()
}

export function connectWebSocket(onMessage) {
    const ws = new WebSocket(WS_URL)
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data)
        onMessage(data)
    }
    ws.onclose = () => {
        // Auto-reconnect after 3 seconds
        setTimeout(() => connectWebSocket(onMessage), 3000)
    }
    ws.onerror = () => ws.close()
    return ws
}

export function getHealthAdvice(aqi) {
    if (aqi <= 120) return {
        icon: '😊', title: 'Air quality is Good',
        text: 'Air quality is satisfactory. Enjoy outdoor activities freely.',
        bg: 'rgba(34, 197, 94, 0.08)', border: '#22c55e'
    }
    if (aqi <= 250) return {
        icon: '😐', title: 'Air quality is Moderate',
        text: 'Sensitive individuals should consider limiting prolonged outdoor exertion.',
        bg: 'rgba(234, 179, 8, 0.08)', border: '#eab308'
    }
    if (aqi <= 350) return {
        icon: '🤢', title: 'Air quality is Unhealthy',
        text: 'Everyone may experience health effects. Limit outdoor exertion.',
        bg: 'rgba(239, 68, 68, 0.08)', border: '#ef4444'
    }
    if (aqi <= 400) return {
        icon: '🚨', title: 'Very Unhealthy Air',
        text: 'Health alert: everyone may experience serious health effects. Stay indoors.',
        bg: 'rgba(168, 85, 250, 0.08)', border: '#a855f7'
    }
    return {
        icon: '☠️', title: 'HAZARDOUS',
        text: 'Health warnings of emergency conditions. Everyone should avoid all outdoor activities.',
        bg: 'rgba(153, 27, 27, 0.12)', border: '#991b1b'
    }
}
