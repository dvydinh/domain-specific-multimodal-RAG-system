import { useEffect } from 'react'

/**
 * Modal popup showing source text and image for a citation.
 * Clicking the overlay or pressing Escape closes it.
 */
export default function CitationPopup({ citation, onClose }) {
  useEffect(() => {
    const handleEsc = (e) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleEsc)
    return () => window.removeEventListener('keydown', handleEsc)
  }, [onClose])

  return (
    <div className="citation-overlay" onClick={onClose}>
      <div
        className="citation-popup"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        id="citation-popup"
      >
        <div className="citation-popup__header">
          <div className="citation-popup__title">
            Source [{citation.id}]
          </div>
          <button
            className="citation-popup__close"
            onClick={onClose}
            aria-label="Close citation"
          >
            ✕
          </button>
        </div>

        {citation.recipe_name && (
          <div className="citation-popup__recipe">
            {citation.recipe_name}
          </div>
        )}

        <div className="citation-popup__text">
          {citation.text}
        </div>

        {citation.image_url && (
          <img
            className="citation-popup__image"
            src={citation.image_url}
            alt={`Source image for ${citation.recipe_name || 'recipe'}`}
            loading="lazy"
          />
        )}

        <div className="citation-popup__source">
          {citation.source_pdf && (
            <span>{citation.source_pdf}</span>
          )}
          {citation.page_number && (
            <span>Page {citation.page_number}</span>
          )}
        </div>
      </div>
    </div>
  )
}
