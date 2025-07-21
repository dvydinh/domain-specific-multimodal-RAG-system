import { useState, useRef, useEffect } from 'react'
import MessageBubble from './MessageBubble'

const EXAMPLE_QUERIES = [
  "Find me a spicy Japanese recipe with pork but without scallion",
  "Show me vegan Italian pasta recipes with images",
  "How do I make Tonkotsu Ramen from scratch?",
  "List all recipes that contain tofu and are gluten-free",
]

export default function ChatInterface({ messages, isLoading, onSend }) {
  const [input, setInput] = useState('')
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

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

  const isEmpty = messages.length === 0

  return (
    <div className="chat-container">
      <div className={`chat-messages ${isEmpty ? 'chat-messages--empty' : ''}`}>
        {isEmpty ? (
          <div className="welcome">
            <div className="welcome__icon">🔍</div>
            <h1 className="welcome__title">What would you like to cook?</h1>
            <p className="welcome__desc">
              Ask me about recipes, ingredients, cooking techniques, or dietary
              preferences. I search across a knowledge graph and vector database
              to give you precise, cited answers.
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
                <div className="message__avatar">🍜</div>
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
            ➤
          </button>
        </form>
      </div>
    </div>
  )
}
