import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
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

    if (!message.content) {
      return (
        <div className="loading-dots">
          <span></span>
          <span></span>
          <span></span>
        </div>
      )
    }

    // Preprocess citation markers [1], [2], etc., so ReactMarkdown parses them as links
    const processedContent = message.content.replace(/\[(\d+)\]/g, '[$1](cit:$1)')

    return (
      <ReactMarkdown 
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ node, ...props }) => {
            if (props.href?.startsWith('cit:')) {
              const citId = props.href.split(':')[1]
              const hasCitation = message.citations && message.citations[citId]
              
              let titleText = `Reference [${citId}]`
              if (hasCitation) {
                if (hasCitation.source_pdf) {
                  titleText = `${hasCitation.source_pdf}${hasCitation.page_number ? ` (Trang ${hasCitation.page_number})` : ''}`
                } else {
                  titleText = `View source [${citId}]`
                }
              }

              return (
                <span
                  className="citation-ref"
                  onClick={(e) => {
                    e.preventDefault()
                    if (hasCitation) setActiveCitation(citId)
                  }}
                  title={titleText}
                  style={{ opacity: hasCitation ? 1 : 0.5, cursor: 'pointer', color: 'var(--primary-color)' }}
                >
                  [{citId}]
                </span>
              )
            }
            return <a {...props} />
          }
        }}
      >
        {processedContent}
      </ReactMarkdown>
    )
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
          {isUser ? (
            <img src="/cherry.png" alt="User" style={{ width: '100%', height: '100%', borderRadius: '50%', objectFit: 'cover' }} />
          ) : (
            <img src="/strawberry.png" alt="AI" style={{ width: '100%', height: '100%', borderRadius: '50%', objectFit: 'cover' }} />
          )}
        </div>
        <div>
          <div className="message__content">
            {renderContent()}
          </div>
          <div className="message__meta">
            <span>{message.timestamp}</span>
            {getQueryTypeBadge()}
            {!isUser && message.graphCount > 0 && (
              <span>{message.graphCount} graph</span>
            )}
            {!isUser && message.vectorCount > 0 && (
              <span>{message.vectorCount} vector</span>
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
