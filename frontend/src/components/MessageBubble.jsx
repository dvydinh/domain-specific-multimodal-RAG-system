import { useState } from 'react'
import CitationPopup from './CitationPopup'

/**
 * Renders a single message bubble (user or assistant).
 * For assistant messages, parses [n] citation markers and makes them clickable.
 */
export default function MessageBubble({ message }) {
  const [activeCitation, setActiveCitation] = useState(null)
  const isUser = message.role === 'user'

  const renderContent = () => {
    if (isUser) return message.content

    // Parse citation markers [1], [2], etc. and make them interactive
    const parts = message.content.split(/(\[\d+\])/g)

    return parts.map((part, idx) => {
      const match = part.match(/^\[(\d+)\]$/)
      if (match) {
        const citId = match[1]
        const hasCitation = message.citations && message.citations[citId]
        return (
          <span
            key={idx}
            className="citation-ref"
            onClick={() => hasCitation && setActiveCitation(citId)}
            title={hasCitation ? `View source [${citId}]` : `Reference [${citId}]`}
            style={{ opacity: hasCitation ? 1 : 0.5 }}
          >
            {citId}
          </span>
        )
      }
      return <span key={idx}>{part}</span>
    })
  }

  const getQueryTypeBadge = () => {
    if (isUser || !message.queryType) return null

    const badges = {
      graph_only: { label: 'Graph', className: 'message__badge--graph' },
      vector_only: { label: 'Vector', className: 'message__badge--vector' },
      hybrid: { label: 'Hybrid', className: 'message__badge--hybrid' },
    }

    const badge = badges[message.queryType]
    if (!badge) return null

    return (
      <span className={`message__badge ${badge.className}`}>
        {badge.label}
      </span>
    )
  }

  return (
    <>
      <div className={`message message--${message.role}`}>
        <div className="message__avatar">
          {isUser ? '👤' : '🍜'}
        </div>
        <div>
          <div className="message__content">
            {renderContent()}
          </div>
          <div className="message__meta">
            <span>{message.timestamp}</span>
            {getQueryTypeBadge()}
            {!isUser && message.graphCount > 0 && (
              <span>📊 {message.graphCount} graph</span>
            )}
            {!isUser && message.vectorCount > 0 && (
              <span>🔍 {message.vectorCount} vector</span>
            )}
          </div>
        </div>
      </div>

      {activeCitation && message.citations[activeCitation] && (
        <CitationPopup
          citation={message.citations[activeCitation]}
          onClose={() => setActiveCitation(null)}
        />
      )}
    </>
  )
}
