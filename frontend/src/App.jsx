import { useState, useRef, useEffect } from 'react'
import ChatInterface from './components/ChatInterface'

const API_BASE = import.meta.env.VITE_API_URL || '/api'

export default function App() {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [uploadToast, setUploadToast] = useState(null)
  
  const [dbFiles, setDbFiles] = useState([])
  const [processingFiles, setProcessingFiles] = useState([])

  const fetchDbFiles = async () => {
    try {
      const res = await fetch(`${API_BASE}/files`)
      const data = await res.json()
      setDbFiles(data.files || [])
    } catch (err) {
      console.error('Failed to fetch files:', err)
    }
  }

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', 'dark')
    
    // Auto-clear dynamic knowledge on reload
    fetch(`${API_BASE}/reset`, { method: 'DELETE' })
      .then(res => res.json())
      .then(data => {
        console.log('Auto-reset:', data.message)
        setProcessingFiles([])
        fetchDbFiles()
      })
      .catch(err => console.error('Failed to auto-reset:', err))
  }, [])

  // Polling for processing files
  useEffect(() => {
    if (processingFiles.length === 0) return
    
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/files`)
        const data = await res.json()
        const currentFiles = data.files || []
        setDbFiles(currentFiles)
        
        // Remove files from processing if they are now in dbFiles
        setProcessingFiles(prev => prev.filter(f => !currentFiles.includes(f)))
      } catch (err) {
        console.error('Polling error:', err)
      }
    }, 3000)
    
    return () => clearInterval(interval)
  }, [processingFiles])

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

    setProcessingFiles(prev => [...prev, file.name])

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: form,
      })
      const data = await res.json()
      setUploadToast(data.message || 'Upload accepted')
    } catch (err) {
      setUploadToast(`Upload failed: ${err.message}`)
      setProcessingFiles(prev => prev.filter(f => f !== file.name))
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

      <div className="main-content">
        <aside className="sidebar">
          <div className="sidebar__header">
            <h3>Knowledge Base</h3>
          </div>
          <div className="sidebar__list">
            {dbFiles.map(f => (
              <div key={f} className="file-item file-item--ready">
                <span className="file-icon">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                    <line x1="16" y1="13" x2="8" y2="13"></line>
                    <line x1="16" y1="17" x2="8" y2="17"></line>
                    <polyline points="10 9 9 9 8 9"></polyline>
                  </svg>
                </span>
                <span className="file-name" title={f}>{f}</span>
                <span className="file-status" title="Ready">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#FFC107" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="20 6 9 17 4 12"></polyline>
                  </svg>
                </span>
              </div>
            ))}
            {processingFiles.map(f => (
              <div key={f} className="file-item file-item--processing">
                <span className="file-icon">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                    <line x1="16" y1="13" x2="8" y2="13"></line>
                    <line x1="16" y1="17" x2="8" y2="17"></line>
                    <polyline points="10 9 9 9 8 9"></polyline>
                  </svg>
                </span>
                <span className="file-name" title={f}>{f}</span>
                <span className="file-status" title="Processing...">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--accent-primary)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="svg-loading-icon">
                    <line x1="12" y1="2" x2="12" y2="6"></line>
                    <line x1="12" y1="18" x2="12" y2="22"></line>
                    <line x1="4.93" y1="4.93" x2="7.76" y2="7.76"></line>
                    <line x1="16.24" y1="16.24" x2="19.07" y2="19.07"></line>
                    <line x1="2" y1="12" x2="6" y2="12"></line>
                    <line x1="18" y1="12" x2="22" y2="12"></line>
                    <line x1="4.93" y1="19.07" x2="7.76" y2="16.24"></line>
                    <line x1="16.24" y1="7.76" x2="19.07" y2="4.93"></line>
                  </svg>
                </span>
              </div>
            ))}
            {dbFiles.length === 0 && processingFiles.length === 0 && (
              <div className="sidebar__empty">No files loaded.</div>
            )}
          </div>
        </aside>

        <main className="chat-main">
          <ChatInterface
            messages={messages}
            isLoading={isLoading}
            onSend={handleSend}
            onUpload={handleUpload}
          />
        </main>
      </div>

      {uploadToast && (
        <div className="upload-toast">{uploadToast}</div>
      )}
    </div>
  )
}
