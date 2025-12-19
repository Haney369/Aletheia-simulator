import React from 'react'

export default function Dashboard({ personas }) {
  return (
    <div className="dashboard">
      <h2>Select Persona</h2>
      <div className="personas-grid">
        {personas.map(p => (
          <div key={p.id} className="persona-card">
            <div className="emoji">{p.emoji}</div>
            <h3>{p.name}</h3>
            <p>{p.description}</p>
          </div>
        ))}
      </div>
    </div>
  )
}