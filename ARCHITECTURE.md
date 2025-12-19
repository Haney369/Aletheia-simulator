# ALETHEIA MVP - SYSTEM ARCHITECTURE

## 📊 EXECUTIVE SUMMARY

**Aletheia** is a Bayesian decision intelligence system powered by:
- **Gemini API** for LLM-based decision analysis
- **MCMC (Markov Chain Monte Carlo)** for posterior sampling
- **Flask backend** for inference serving
- **React frontend** for interactive decision exploration

**Deployment:** Google Cloud Run (auto-scaling)  
**Latency:** ~2-5 seconds per analysis (LLM + MCMC)  
**Scalability:** 0-100 instances (on-demand)

---
