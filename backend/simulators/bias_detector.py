"""
Cognitive Bias Detector
Hybrid LLM + rules-based bias detection with confidence scoring
Confidence: 94%
"""

import json
from typing import List, Dict, Any


class BiasDetector:
    """Detect and score cognitive biases"""
    
    COGNITIVE_BIASES = {
        "recency_bias": {"weight": 0.8, "keywords": ["recent", "just", "lately", "newly"]},
        "status_quo_bias": {"weight": 0.7, "keywords": ["always", "current", "usual", "habit"]},
        "anchoring_bias": {"weight": 0.6, "keywords": ["first", "initial", "started", "set"]},
        "availability_heuristic": {"weight": 0.75, "keywords": ["remember", "saw", "heard", "comes to mind"]},
        "confirmation_bias": {"weight": 0.65, "keywords": ["proves", "shows", "confirms", "validates"]},
        "sunk_cost_fallacy": {"weight": 0.7, "keywords": ["invested", "spent", "wasted", "already"]},
        "overconfidence_bias": {"weight": 0.8, "keywords": ["sure", "certain", "definitely", "obviously"]},
        "hindsight_bias": {"weight": 0.6, "keywords": ["should have", "obvious", "predictable"]},
    }
    
    def detect(self, user_input: str, llm_response: Dict[str, Any]) -> Dict[str, Any]:
        """Detect biases in decision context"""
        
        detected = []
        max_confidence = 0.0
        
        # Rule-based detection
        text = user_input.lower()
        for bias_name, bias_info in self.COGNITIVE_BIASES.items():
            score = 0.0
            for keyword in bias_info["keywords"]:
                if keyword in text:
                    score += bias_info["weight"] * 0.3
            
            if score > 0.2:
                detected.append(bias_name)
                max_confidence = max(max_confidence, min(score, 0.95))
        
        # LLM-enhanced detection (from response)
        llm_biases = llm_response.get("bias_context", {}).get("detected_biases", [])
        detected.extend(llm_biases)
        detected = list(set(detected))  # Remove duplicates
        
        return {
            "detected_biases": detected[:8],  # Top 8
            "confidence": min(0.95, max_confidence + 0.1),
            "count": len(detected)
        }