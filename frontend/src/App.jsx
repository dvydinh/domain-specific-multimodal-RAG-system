import { useState, useRef, useEffect } from 'react'
import ChatInterface from './components/ChatInterface'

const API_BASE = '/api'

export default function App() {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [uploadToast, setUploadToast] = useState(null)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', 'dark')
  }, [])

  const handleSend = async (question) => {
    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: question,
      timestamp: new Date().toLocaleTimeString(),
    }
    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)

    // Create a placeholder assistant message for streaming
    const assistantId = Date.now() + 1
    const assistantMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      citations: {},
      queryType: '',
      graphCount: 0,
      vectorCount: 0,
      timestamp: new Date().toLocaleTimeString(),
    }
    setMessages(prev => [...prev, assistantMessage])

    try {
      const res = await fetch(`${API_BASE}/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          include_images: true,
          top_k: 5,
        }),
      })

      if (!res.ok) throw new Error(`API error: ${res.status}`)

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const event = JSON.parse(line.slice(6))

            if (event.event === 'metadata') {
              setMessages(prev => prev.map(m =>
                m.id === assistantId
                  ? { ...m, queryType: event.data.query_type, graphCount: event.data.graph_results_count, vectorCount: event.data.vector_results_count }
                  : m
              ))
            } else if (event.event === 'token') {
              setMessages(prev => prev.map(m =>
                m.id === assistantId
                  ? { ...m, content: m.content + event.data }
                  : m
              ))
            } else if (event.event === 'done') {
              setMessages(prev => prev.map(m =>
                m.id === assistantId
                  ? { ...m, citations: event.data.citations || {} }
                  : m
              ))
            }
          } catch (parseErr) {
            // Skip malformed SSE lines
          }
        }
      }
    } catch (err) {
      setMessages(prev => prev.map(m =>
        m.id === assistantId
          ? { ...m, content: `Something went wrong: ${err.message}. Make sure the backend server is running.`, queryType: 'error' }
          : m
      ))
    } finally {
      setIsLoading(false)
    }
  }

  const handleUpload = async (file) => {
    if (!file || !file.name.toLowerCase().endsWith('.pdf')) {
      setUploadToast('Only PDF files are accepted')
      setTimeout(() => setUploadToast(null), 3000)
      return
    }

    const form = new FormData()
    form.append('file', file)

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: form,
      })
      const data = await res.json()
      setUploadToast(data.message || 'Upload accepted')
    } catch (err) {
      setUploadToast(`Upload failed: ${err.message}`)
    }
    setTimeout(() => setUploadToast(null), 3000)
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header__brand">
          <img src="/strawberry.png" alt="Strawberry Logo" className="header__logo" />
          <div>
            <div className="header__title">Recipe RAG</div>
            <div className="header__subtitle">
              Knowledge Graph + Vector Search
            </div>
          </div>
        </div>
        <div className="header__actions">
          <div className="header__status">
            <span className="status-dot" />
            Online
          </div>
        </div>
      </header>

      <ChatInterface
        messages={messages}
        isLoading={isLoading}
        onSend={handleSend}
        onUpload={handleUpload}
      />

      {uploadToast && (
        <div className="upload-toast">{uploadToast}</div>
      )}
    </div>
  )
}
