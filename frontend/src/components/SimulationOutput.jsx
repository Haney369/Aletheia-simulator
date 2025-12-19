import React from 'react'

export default function SimulationOutput({ result, onNewAnalysis }) {
  if (result.error) {
    return (
      <div className="output-error">
        <h2>❌ Error</h2>
        <p>{result.error}</p>
        <button onClick={onNewAnalysis} className="btn-primary">New Analysis</button>
      </div>
    )
  }

  return (
    <div className="simulation-output">
      <div className="result-header">
        <h2>{result.detected_persona?.toUpperCase()}</h2>
        <div className="confidence">
          Confidence: {(result.confidence_score * 100).toFixed(0)}%
        </div>
      </div>

      {result.decision_variables && (
        <div className="metrics">
          <div className="metric">
            <label>Posterior Mean</label>
            <value>{(result.decision_variables.posterior_mean * 100).toFixed(1)}%</value>
          </div>
          <div className="metric">
            <label>Confidence Interval (95%)</label>
            <value>
              [{result.decision_variables.confidence_interval_95?.[0]?.toFixed(2)},
               {result.decision_variables.confidence_interval_95?.[1]?.toFixed(2)}]
            </value>
          </div>
        </div>
      )}

      {result.bias_context && (
        <div className="biases">
          <h3>⚠️ Detected Biases</h3>
          <ul>
            {result.bias_context.detected_biases?.map((b, i) => (
              <li key={i}>{b}</li>
            ))}
          </ul>
        </div>
      )}

      {result.recommendation && (
        <div className="recommendation">
          <h3>💡 Recommendation</h3>
          <p>{result.recommendation}</p>
        </div>
      )}

      <button onClick={onNewAnalysis} className="btn-primary">
        New Analysis
      </button>
    </div>
  )
}