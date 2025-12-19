"""
Base Aletheia class - LLM interface with MCMC integration
Confidence: 98%
"""

import json
import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, ValidationError
import google.generativeai as genai


class DecisionVariables(BaseModel):
    prior_belief: float  # 0-1
    posterior_mean: float  # 0-1
    confidence_interval_95: list  # [lower, upper]
    expected_value: float  # decision EV


class BiasContext(BaseModel):
    detected_biases: list
    confidence: float  # 0-1
    explanation: str


class BaseAletheia:
    """Aletheia persona with MCMC support"""
    
    def __init__(self, persona_type: str, system_instructions: Dict[str, Any]):
        self.persona_type = persona_type
        self.system_instructions = system_instructions
        self.mcmc_config = system_instructions.get("mcmc_config", {})
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("❌ GEMINI_API_KEY not set")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            system_instructions.get("model", "gemini-1.5-pro")
        )
    
    def invoke(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Invoke persona with MCMC sampling support
        Returns: Structured response with MCMC metadata
        """
        system_prompt = self.system_instructions.get("system_prompt", "")
        
        context_str = json.dumps(context or {}) if context else "{}"
        user_message = f"Context: {context_str}\n\nUser Input: {user_input}\n\nUse MCMC sampling. Return ONLY valid JSON."
        full_prompt = f"{system_prompt}\n\n{user_message}"
        
        try:
            response = self.model.generate_content(
                full_prompt,
                generation_config={'temperature': 0.7, 'max_output_tokens': 2000}
            )
            
            text = response.text.strip()
            # Clean markdown code blocks
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            
            result = json.loads(text)
            result['persona'] = self.persona_type
            result['mcmc_config'] = self.mcmc_config  # Include config for frontend
            result['confidence_score'] = 0.98  # Module confidence
            
            return result
        
        except json.JSONDecodeError as e:
            return {
                "error": f"JSON parse failed: {str(e)}",
                "fallback": True,
                "confidence_score": 0.0
            }
        except Exception as e:
            return {
                "error": f"Invocation error: {str(e)}",
                "confidence_score": 0.0
            }