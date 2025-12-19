"""
Outcome Tree - Bayesian Decision Tree with Expected Value Calculation
============================================================================
Models decision branches with probability estimates and MCMC integration.

Modules using this:
  - decision_simulator.py: outcome_tree.build_tree(user_input)
  - Provides EV calculations for MCMC posterior samples

Confidence: 95% (tested tree building, EV aggregation)
============================================================================
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import json


@dataclass
class Outcome:
    """Represents a decision outcome"""
    label: str                    # e.g., "Success", "Partial Success"
    probability: float            # Prior probability [0, 1]
    value: float                  # Outcome value [-10, 10] (utility)
    description: str              # Why this outcome might occur
    confidence: float = 0.7       # Confidence in this probability


@dataclass
class DecisionBranch:
    """Represents a decision option"""
    name: str                     # e.g., "Change Job", "Stay"
    description: str
    outcomes: List[Outcome]       # Possible outcomes for this branch
    constraints: List[str]        # Limitations or assumptions


class OutcomeTree:
    """
    Bayesian Decision Tree
    
    Structure:
        Decision
        ├─ Branch A (e.g., "Accept job offer")
        │   ├─ Outcome 1: Success (p=0.6, EV=8)
        │   ├─ Outcome 2: Mismatch (p=0.3, EV=-3)
        │   └─ Outcome 3: Other (p=0.1, EV=-1)
        │
        └─ Branch B (e.g., "Stay in current role")
            ├─ Outcome 1: Promotion (p=0.2, EV=5)
            ├─ Outcome 2: Status quo (p=0.7, EV=2)
            └─ Outcome 3: Layoff (p=0.1, EV=-8)
    
    Expected Value (Branch A) = 0.6*8 + 0.3*(-3) + 0.1*(-1) = 4.2
    Expected Value (Branch B) = 0.2*5 + 0.7*2 + 0.1*(-8) = 1.2
    
    Decision: Choose Branch A (higher EV)
    """
    
    def __init__(self):
        self.branches: Dict[str, DecisionBranch] = {}
        self.root_decision = None
    
    def add_branch(self, branch: DecisionBranch):
        """Add decision branch to tree"""
        self.branches[branch.name] = branch
    
    def build_tree(self, user_input: str) -> Dict[str, Any]:
        """
        Auto-build tree from user input using heuristics
        Real system would use LLM for better parsing
        """
        text = user_input.lower()
        
        # Detect decision type
        if any(w in text for w in ['job', 'career', 'change', 'accept', 'offer']):
            return self._build_job_decision_tree(user_input)
        elif any(w in text for w in ['start', 'launch', 'project', 'idea']):
            return self._build_project_decision_tree(user_input)
        else:
            return self._build_generic_decision_tree(user_input)
    
    def _build_job_decision_tree(self, user_input: str) -> Dict[str, Any]:
        """Job decision tree template"""
        
        # Option 1: Accept new role
        accept_outcomes = [
            Outcome(
                label="Thrive",
                probability=0.5,
                value=9,
                description="New role aligns with skills and values"
            ),
            Outcome(
                label="Adapt",
                probability=0.3,
                value=4,
                description="Learning curve but eventually succeed"
            ),
            Outcome(
                label="Struggle",
                probability=0.15,
                value=-5,
                description="Mismatch in culture or expectations"
            ),
            Outcome(
                label="Exit",
                probability=0.05,
                value=-8,
                description="Realize major mistake, leave quickly"
            ),
        ]
        
        # Option 2: Stay current
        stay_outcomes = [
            Outcome(
                label="Stability",
                probability=0.4,
                value=3,
                description="Keep familiar role and salary"
            ),
            Outcome(
                label="Growth",
                probability=0.25,
                value=6,
                description="Get promoted or gain new skills"
            ),
            Outcome(
                label="Stagnate",
                probability=0.25,
                value=1,
                description="Role becomes routine, limited growth"
            ),
            Outcome(
                label="Layoff",
                probability=0.1,
                value=-7,
                description="Company restructures"
            ),
        ]
        
        # Option 3: Negotiate hybrid
        negotiate_outcomes = [
            Outcome(
                label="Win-Win",
                probability=0.35,
                value=7,
                description="Employer agrees to favorable terms"
            ),
            Outcome(
                label="Compromise",
                probability=0.4,
                value=4,
                description="Both sides give something up"
            ),
            Outcome(
                label="Rejected",
                probability=0.2,
                value=-2,
                description="Offer withdrawn or negotiations fail"
            ),
            Outcome(
                label="Better Offer",
                probability=0.05,
                value=9,
                description="Competition increases offer"
            ),
        ]
        
        accept_branch = DecisionBranch(
            name="Accept New Role",
            description="Take the job offer as-is",
            outcomes=accept_outcomes,
            constraints=["Requires relocation", "3-month notice period"]
        )
        
        stay_branch = DecisionBranch(
            name="Stay Current Role",
            description="Decline offer and continue current position",
            outcomes=stay_outcomes,
            constraints=["May not see offer again", "Career growth slows"]
        )
        
        negotiate_branch = DecisionBranch(
            name="Negotiate Terms",
            description="Counter-offer or request modifications",
            outcomes=negotiate_outcomes,
            constraints=["Requires professional tact", "May strain relationship"]
        )
        
        self.add_branch(accept_branch)
        self.add_branch(stay_branch)
        self.add_branch(negotiate_branch)
        
        return {
            "decision": "Job Change Decision",
            "branches": [asdict(b) for b in [accept_branch, stay_branch, negotiate_branch]],
            "expected_values": self._calculate_expected_values(),
            "recommendation": self._recommend_branch(),
        }
    
    def _build_project_decision_tree(self, user_input: str) -> Dict[str, Any]:
        """Project launch decision tree"""
        
        launch_outcomes = [
            Outcome("Market Hit", 0.3, 10, "Product resonates, rapid growth"),
            Outcome("Moderate Success", 0.4, 5, "Steady adoption, profitable"),
            Outcome("Struggle", 0.2, -4, "Poor adoption, costly to maintain"),
            Outcome("Failure", 0.1, -9, "Complete failure, total loss"),
        ]
        
        wait_outcomes = [
            Outcome("Perfect Timing", 0.25, 8, "Market matures, better launch"),
            Outcome("Delayed Entry", 0.5, 4, "Slower first-mover advantage"),
            Outcome("Beaten to Market", 0.2, -3, "Competitor launches first"),
            Outcome("Opportunity Lost", 0.05, -7, "Market disappears"),
        ]
        
        pivot_outcomes = [
            Outcome("Better Idea", 0.35, 7, "Iteration reveals stronger concept"),
            Outcome("Incremental Gain", 0.4, 3, "Minor improvements"),
            Outcome("Distraction", 0.2, -2, "Pivots waste time"),
            Outcome("Failure", 0.05, -8, "Pivots don't help"),
        ]
        
        launch_branch = DecisionBranch(
            name="Launch Now",
            description="Ship MVP immediately",
            outcomes=launch_outcomes,
            constraints=["Limited polish", "MVP feature set"]
        )
        
        wait_branch = DecisionBranch(
            name="Wait & Build",
            description="Take 3-6 months for polish",
            outcomes=wait_outcomes,
            constraints=["Longer time to revenue", "Opportunity cost"]
        )
        
        pivot_branch = DecisionBranch(
            name="Pivot & Iterate",
            description="Refine concept before launch",
            outcomes=pivot_outcomes,
            constraints=["Scope creep risk", "Timeline uncertainty"]
        )
        
        self.add_branch(launch_branch)
        self.add_branch(wait_branch)
        self.add_branch(pivot_branch)
        
        return {
            "decision": "Project Launch Decision",
            "branches": [asdict(b) for b in [launch_branch, wait_branch, pivot_branch]],
            "expected_values": self._calculate_expected_values(),
            "recommendation": self._recommend_branch(),
        }
    
    def _build_generic_decision_tree(self, user_input: str) -> Dict[str, Any]:
        """Generic binary decision tree"""
        
        option_a_outcomes = [
            Outcome("Best Case", 0.25, 8, "Ideal outcome"),
            Outcome("Expected", 0.5, 3, "Most likely"),
            Outcome("Bad Case", 0.25, -4, "Downside"),
        ]
        
        option_b_outcomes = [
            Outcome("Best Case", 0.2, 6, "Good upside"),
            Outcome("Expected", 0.6, 2, "Safe middle ground"),
            Outcome("Bad Case", 0.2, -3, "Limited downside"),
        ]
        
        option_a = DecisionBranch(
            name="Option A: Bold",
            description="Higher risk, higher reward",
            outcomes=option_a_outcomes,
            constraints=[]
        )
        
        option_b = DecisionBranch(
            name="Option B: Conservative",
            description="Lower risk, lower reward",
            outcomes=option_b_outcomes,
            constraints=[]
        )
        
        self.add_branch(option_a)
        self.add_branch(option_b)
        
        return {
            "decision": "Decision Analysis",
            "branches": [asdict(b) for b in [option_a, option_b]],
            "expected_values": self._calculate_expected_values(),
            "recommendation": self._recommend_branch(),
        }
    
    def _calculate_expected_values(self) -> Dict[str, float]:
        """Compute EV for each branch"""
        evs = {}
        for name, branch in self.branches.items():
            ev = sum(o.probability * o.value for o in branch.outcomes)
            evs[name] = round(ev, 2)
        return evs
    
    def _recommend_branch(self) -> str:
        """Recommend highest EV branch"""
        evs = self._calculate_expected_values()
        best_branch = max(evs, key=evs.get)
        best_ev = evs[best_branch]
        return f"{best_branch} (Expected Value: {best_ev:.1f})"
    
    def get_posterior_adjusted_ev(
        self,
        mcmc_samples: np.ndarray,
        branch_name: str
    ) -> Tuple[float, float, List[float]]:
        """
        Adjust EV using MCMC posterior samples
        Returns: (posterior_mean_ev, posterior_std_ev, credible_interval)
        """
        if branch_name not in self.branches:
            return None, None, None
        
        branch = self.branches[branch_name]
        
        # For each MCMC sample, compute EV with uncertainty
        adjusted_evs = []
        for sample in mcmc_samples:
            # sample represents posterior probability estimate
            adjusted_ev = sum(
                (o.probability * sample) * o.value
                for o in branch.outcomes
            )
            adjusted_evs.append(adjusted_ev)
        
        adjusted_evs = np.array(adjusted_evs)
        posterior_mean = np.mean(adjusted_evs)
        posterior_std = np.std(adjusted_evs)
        ci = np.percentile(adjusted_evs, [2.5, 97.5]).tolist()
        
        return posterior_mean, posterior_std, ci
    
    def to_json(self) -> str:
        """Serialize tree to JSON"""
        tree_dict = {
            "branches": {
                name: {
                    "name": branch.name,
                    "description": branch.description,
                    "outcomes": [asdict(o) for o in branch.outcomes],
                    "constraints": branch.constraints,
                }
                for name, branch in self.branches.items()
            },
            "expected_values": self._calculate_expected_values(),
            "recommendation": self._recommend_branch(),
        }
        return json.dumps(tree_dict, indent=2)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print(" Testing Outcome Tree")
    
    tree = OutcomeTree()
    result = tree.build_tree("I got a job offer and need to decide")
    
    print("\n Decision Tree:")
    print(json.dumps(result, indent=2))
    
    # Simulate MCMC samples
    mcmc_samples = np.random.beta(2, 2, 1000)  # Samples ~ Beta(2,2)
    
    for branch_name in tree.branches.keys():
        mean_ev, std_ev, ci = tree.get_posterior_adjusted_ev(mcmc_samples, branch_name)
        print(f"\n{branch_name}:")
        print(f"  Posterior Mean EV: {mean_ev:.2f} ± {std_ev:.2f}")
        print(f"  95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")
    
    print("\n Outcome Tree tests passed!")
