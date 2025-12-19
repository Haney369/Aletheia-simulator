import React from 'react'

export default function TimelineSimulator({ trajectory, title }) {
  return (
    <div className="timeline">
      <h3>{title}</h3>
      <div className="trajectory-chart">
        {trajectory && trajectory.map((val, i) => (
          <div
            key={i}
            className="bar"
            style={{ height: `${val * 100}%` }}
            title={`Hour ${i}: ${(val * 100).toFixed(0)}%`}
          />
        ))}
      </div>
    </div>
  )
}