import React from 'react'

export default function WizardFlow({
  personas,
  selectedPersona,
  onPersonaSelect,
  onAnalyze,
  loading,
  input,
  onInputChange
}) {
  return (
    <div className="wizard-flow">
      <div className="input-section">
        <textarea
          value={input}
          onChange={(e) => onInputChange(e.target.value)}
          placeholder="Your challenge..."
          rows={4}
        />
      </div>

      <div className="personas-section">
        <h3>Select Analysis Type</h3>
        <div className="personas-grid">
          {personas.map(p => (
            <div
              key={p.id}
              className={`persona-card ${selectedPersona === p.id ? 'active' : ''}`}
              onClick={() => onPersonaSelect(p.id)}
            >
              <div className="emoji">{p.emoji}</div>
              <h4>{p.name}</h4>
            </div>
          ))}
        </div>
      </div>

      <button
        onClick={onAnalyze}
        disabled={loading || !input.trim()}
        className="btn-primary"
      >
        {loading ? 'Analyzing...' : 'Analyze with MCMC'}
      </button>
    </div>
  )
}