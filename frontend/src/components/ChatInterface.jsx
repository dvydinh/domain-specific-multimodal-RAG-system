import { useState, useRef, useEffect } from 'react'
import MessageBubble from './MessageBubble'

const EXAMPLE_QUERIES = [
  "Find me a spicy Japanese recipe with pork but without scallion",
  "Show me vegan Italian pasta recipes with images",
  "How do I make Tonkotsu Ramen from scratch?",
  "List all recipes that contain tofu and are gluten-free",
]

export default function ChatInterface({ messages, isLoading, onSend, onUpload }) {
  const [input, setInput] = useState('')
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)
  const fileInputRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  const handleSubmit = (e) => {
    e.preventDefault()
    const text = input.trim()
    if (!text || isLoading) return
    onSend(text)
    setInput('')
  }

  const handleExampleClick = (example) => {
    if (isLoading) return
    onSend(example)
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  const handleFileChange = (e) => {
    const file = e.target.files?.[0]
    if (file) {
      onUpload(file)
      e.target.value = ''
    }
  }

  const isEmpty = messages.length === 0

  return (
    <div className="chat-container">
      <div className={`chat-messages ${isEmpty ? 'chat-messages--empty' : ''}`}>
        {isEmpty ? (
          <div className="welcome">
            <div className="welcome__visual" />
            <h1 className="welcome__title">What would you like to cook?</h1>
            <p className="welcome__desc">
              Ask about recipes, ingredients, or cooking techniques.
              Answers are sourced from a knowledge graph and vector database
              with precise citations.
            </p>
            <div className="welcome__examples">
              {EXAMPLE_QUERIES.map((q, i) => (
                <button
                  key={i}
                  className="welcome__example"
                  onClick={() => handleExampleClick(q)}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))}
            {isLoading && (
              <div className="message message--assistant">
                <div className="message__avatar">
                  <img src="/strawberry.png" alt="AI" style={{ width: '100%', height: '100%', borderRadius: '50%', objectFit: 'cover' }} />
                </div>
                <div className="message__content">
                  <div className="loading-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-area">
        <form className="chat-input-form" onSubmit={handleSubmit}>
          <div className="upload-btn" title="Upload a PDF cookbook">
            <span>+</span>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              onChange={handleFileChange}
            />
          </div>
          <textarea
            ref={inputRef}
            className="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about recipes, ingredients, or cooking techniques..."
            rows={1}
            disabled={isLoading}
            id="chat-input"
          />
          <button
            type="submit"
            className="chat-submit"
            disabled={isLoading || !input.trim()}
            id="chat-submit"
          >
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" style={{ transform: 'translateX(-1px) translateY(1px)' }}>
              <line x1="22" y1="2" x2="11" y2="13"></line>
              <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
            </svg>
          </button>
        </form>
      </div>
    </div>
  )
}
