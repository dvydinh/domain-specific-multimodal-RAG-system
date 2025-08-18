import { useState, useRef, useEffect } from 'react'
import ChatInterface from './components/ChatInterface'

const API_BASE = '/api'

export default function App() {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('rag-theme') || 'light'
  })
  const [uploadToast, setUploadToast] = useState(null)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('rag-theme', theme)
  }, [theme])

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light')
  }

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
        content: `Something went wrong: ${err.message}. Make sure the backend server is running.`,
        citations: {},
        queryType: 'error',
        timestamp: new Date().toLocaleTimeString(),
      }
      setMessages(prev => [...prev, errorMessage])
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
          <button
            className="theme-toggle"
            onClick={toggleTheme}
            aria-label="Toggle theme"
            title={theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}
          >
            <span className="theme-toggle__knob" />
          </button>
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
