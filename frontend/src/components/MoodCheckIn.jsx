import React from 'react'

export default function MoodCheckIn({ onNext, onInputChange, input }) {
  return (
    <div className="mood-checkin">
      <div className="checkin-card">
        <h2>How are you feeling today?</h2>
        <textarea
          placeholder="Describe your challenge, decision, or situation..."
          value={input}
          onChange={(e) => onInputChange(e.target.value)}
          rows={6}
        />
        <button onClick={onNext} className="btn-primary">
          Continue →
        </button>
      </div>
    </div>
  )
}