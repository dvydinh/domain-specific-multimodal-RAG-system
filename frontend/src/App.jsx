import { useState, useRef, useEffect } from 'react'
import ChatInterface from './components/ChatInterface'

const API_BASE = '/api'

export default function App() {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)

  const handleSend = async (question) => {
    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: question,
      timestamp: new Date().toLocaleTimeString(),
    }
    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)

    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          include_images: true,
          top_k: 5,
        }),
      })

      if (!res.ok) throw new Error(`API error: ${res.status}`)

      const data = await res.json()

      const assistantMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: data.response,
        citations: data.citations || {},
        queryType: data.query_type,
        graphCount: data.graph_results_count,
        vectorCount: data.vector_results_count,
        timestamp: new Date().toLocaleTimeString(),
      }
      setMessages(prev => [...prev, assistantMessage])
    } catch (err) {
      const errorMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: `Sorry, something went wrong: ${err.message}. Make sure the backend is running.`,
        citations: {},
        queryType: 'error',
        timestamp: new Date().toLocaleTimeString(),
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header__brand">
          <div className="header__icon">🍜</div>
          <div>
            <div className="header__title">Recipe RAG</div>
            <div className="header__subtitle">
              Multimodal Knowledge Graph + Vector Search
            </div>
          </div>
        </div>
        <div className="header__status">
          <span className="status-dot" />
          System Online
        </div>
      </header>

      <ChatInterface
        messages={messages}
        isLoading={isLoading}
        onSend={handleSend}
      />
    </div>
  )
}
