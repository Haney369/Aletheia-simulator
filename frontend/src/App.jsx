import React, { useState } from 'react'
import axios from 'axios'
import Dashboard from './components/Dashboard'
import MoodCheckIn from './components/MoodCheckIn'
import WizardFlow from './components/WizardFlow'
import SimulationOutput from './components/SimulationOutput'
import './index.css'

function App() {
  const [currentStep, setCurrentStep] = useState('checkin')
  const [userInput, setUserInput] = useState('')
  const [selectedPersona, setSelectedPersona] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [personas, setPersonas] = useState([])

  React.useEffect(() => {
    // Load personas
    axios.get('/api/personas')
      .then(res => setPersonas(res.data.personas))
      .catch(err => console.error(err))
  }, [])

  const handleAnalyze = async () => {
    if (!userInput.trim()) return

    setLoading(true)
    try {
      const response = await axios.post('/api/analyze', {
        input: userInput,
        persona: selectedPersona
      })
      setResult(response.data)
      setCurrentStep('output')
    } catch (error) {
      console.error('Error:', error)
      setResult({ error: error.response?.data?.error || 'Analysis failed' })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>🧠 Aletheia MVP</h1>
        <p>Cognitive Recalibration with MCMC</p>
      </header>

      {currentStep === 'checkin' && (
        <MoodCheckIn
          onNext={() => setCurrentStep('wizard')}
          onInputChange={setUserInput}
          input={userInput}
        />
      )}

      {currentStep === 'wizard' && (
        <WizardFlow
          personas={personas}
          selectedPersona={selectedPersona}
          onPersonaSelect={setSelectedPersona}
          onAnalyze={handleAnalyze}
          loading={loading}
          input={userInput}
          onInputChange={setUserInput}
        />
      )}

      {currentStep === 'output' && result && (
        <SimulationOutput
          result={result}
          onNewAnalysis={() => {
            setCurrentStep('checkin')
            setUserInput('')
            setSelectedPersona(null)
            setResult(null)
          }}
        />
      )}
    </div>
  )
}
export default App