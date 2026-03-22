"""RAG MDP Environment: state, action, reward with cost modeling.
The policy decides when/what/how much to retrieve."""

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class RAGAction(IntEnum):
    NO_RETRIEVE = 0
    RETRIEVE_1 = 1
    RETRIEVE_3 = 2
    RETRIEVE_5 = 3
    RETRIEVE_10 = 4
    REWRITE = 5
    MULTI_HOP = 6


ACTION_COSTS = {
    RAGAction.NO_RETRIEVE: 0.0,
    RAGAction.RETRIEVE_1: 0.1,
    RAGAction.RETRIEVE_3: 0.3,
    RAGAction.RETRIEVE_5: 0.5,
    RAGAction.RETRIEVE_10: 1.0,
    RAGAction.REWRITE: 0.2,
    RAGAction.MULTI_HOP: 1.5,
}

ACTION_K = {
    RAGAction.NO_RETRIEVE: 0,
    RAGAction.RETRIEVE_1: 1,
    RAGAction.RETRIEVE_3: 3,
    RAGAction.RETRIEVE_5: 5,
    RAGAction.RETRIEVE_10: 10,
    RAGAction.REWRITE: 5,
    RAGAction.MULTI_HOP: 5,
}


@dataclass
class RAGState:
    """State representation for the RAG MDP."""
    query: str
    query_embedding: Optional[torch.Tensor] = None
    query_complexity: float = 0.0  # estimated query difficulty [0, 1]
    retrieval_confidence: float = 0.0  # confidence from previous retrieval [0, 1]
    num_retrievals_done: int = 0
    retrieved_docs: List[str] = field(default_factory=list)
    current_answer: Optional[str] = None
    answer_confidence: float = 0.0

    def to_features(self) -> torch.Tensor:
        """Convert state to feature vector for policy input."""
        features = [
            self.query_complexity,
            self.retrieval_confidence,
            float(self.num_retrievals_done) / 5.0,
            self.answer_confidence,
            float(len(self.retrieved_docs)) / 10.0,
        ]
        return torch.tensor(features, dtype=torch.float32)


@dataclass
class RAGTransition:
    """A single MDP transition."""
    state: RAGState
    action: RAGAction
    reward: float
    next_state: RAGState
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


class RAGRewardFunction:
    """Compute reward: accuracy - lambda * cost."""

    def __init__(self, cost_lambda: float = 0.3, accuracy_weight: float = 1.0):
        self.cost_lambda = cost_lambda
        self.accuracy_weight = accuracy_weight

    def compute(self, predicted_answer: str, gold_answer: str, action: RAGAction,
                answer_confidence: float = 0.0) -> Tuple[float, Dict[str, float]]:
        """Compute reward for a RAG action."""
        accuracy = self._compute_accuracy(predicted_answer, gold_answer)
        cost = ACTION_COSTS[action]
        confidence_bonus = 0.1 * answer_confidence if accuracy > 0.5 else 0.0
        reward = self.accuracy_weight * accuracy - self.cost_lambda * cost + confidence_bonus
        info = {
            "accuracy": accuracy,
            "cost": cost,
            "confidence_bonus": confidence_bonus,
            "raw_reward": reward,
        }
        return reward, info

    @staticmethod
    def _compute_accuracy(predicted: str, gold: str) -> float:
        """Simple token-level F1."""
        import re
        def normalize(s):
            s = s.lower()
            s = re.sub(r'\b(a|an|the)\b', ' ', s)
            s = re.sub(r'[^a-z0-9\s]', '', s)
            return ' '.join(s.split())
        pred_tokens = set(normalize(predicted).split())
        gold_tokens = set(normalize(gold).split())
        if not pred_tokens or not gold_tokens:
            return float(pred_tokens == gold_tokens)
        common = pred_tokens & gold_tokens
        if not common:
            return 0.0
        prec = len(common) / len(pred_tokens)
        rec = len(common) / len(gold_tokens)
        return 2 * prec * rec / (prec + rec)


class RAGEnvironment:
    """Full RAG MDP environment."""

    def __init__(self, retriever=None, generator=None, reward_fn: Optional[RAGRewardFunction] = None,
                 max_steps: int = 3):
        self.retriever = retriever
        self.generator = generator
        self.reward_fn = reward_fn or RAGRewardFunction()
        self.max_steps = max_steps
        self.current_state: Optional[RAGState] = None
        self.gold_answer: Optional[str] = None
        self.step_count = 0

    def reset(self, query: str, gold_answer: str, query_embedding: Optional[torch.Tensor] = None) -> RAGState:
        """Reset environment with a new query."""
        complexity = self._estimate_complexity(query)
        self.current_state = RAGState(
            query=query,
            query_embedding=query_embedding,
            query_complexity=complexity,
        )
        self.gold_answer = gold_answer
        self.step_count = 0
        return self.current_state

    def step(self, action: RAGAction) -> RAGTransition:
        """Execute an action in the environment."""
        state = self.current_state
        self.step_count += 1

        # Execute action
        if action == RAGAction.NO_RETRIEVE:
            answer = self._generate_answer(state.query, [])
        elif action == RAGAction.REWRITE:
            rewritten = self._rewrite_query(state.query)
            docs = self._retrieve(rewritten, ACTION_K[action])
            state.retrieved_docs.extend(docs)
            answer = self._generate_answer(state.query, state.retrieved_docs)
        elif action == RAGAction.MULTI_HOP:
            docs1 = self._retrieve(state.query, ACTION_K[action])
            hop2_query = f"{state.query} {' '.join(docs1[:2])}"
            docs2 = self._retrieve(hop2_query, 3)
            state.retrieved_docs.extend(docs1 + docs2)
            answer = self._generate_answer(state.query, state.retrieved_docs)
        else:
            k = ACTION_K[action]
            docs = self._retrieve(state.query, k)
            state.retrieved_docs.extend(docs)
            answer = self._generate_answer(state.query, state.retrieved_docs)

        confidence = self._estimate_confidence(answer)
        reward, info = self.reward_fn.compute(answer, self.gold_answer, action, confidence)

        done = self.step_count >= self.max_steps or action == RAGAction.NO_RETRIEVE

        next_state = RAGState(
            query=state.query,
            query_embedding=state.query_embedding,
            query_complexity=state.query_complexity,
            retrieval_confidence=confidence,
            num_retrievals_done=state.num_retrievals_done + (0 if action == RAGAction.NO_RETRIEVE else 1),
            retrieved_docs=state.retrieved_docs,
            current_answer=answer,
            answer_confidence=confidence,
        )
        self.current_state = next_state

        transition = RAGTransition(
            state=state, action=action, reward=reward, next_state=next_state, done=done, info=info,
        )
        return transition

    def _retrieve(self, query: str, k: int) -> List[str]:
        """Retrieve k documents. Falls back to dummy if no retriever."""
        if self.retriever is not None:
            return self.retriever.search(query, k=k)
        return [f"[Retrieved passage {i+1} for: {query[:50]}...]" for i in range(k)]

    def _generate_answer(self, query: str, context: List[str]) -> str:
        """Generate answer. Falls back to simple heuristic if no generator."""
        if self.generator is not None:
            ctx = "\n".join(context[:5])
            return self.generator.generate(query, ctx)
        if context:
            return f"Based on retrieved information: {context[0][:100]}"
        return f"Direct answer to: {query[:100]}"

    def _rewrite_query(self, query: str) -> str:
        """Rewrite query for better retrieval."""
        return f"Detailed information about: {query}"

    @staticmethod
    def _estimate_complexity(query: str) -> float:
        """Heuristic query complexity estimation."""
        complexity = 0.0
        complexity += min(len(query.split()) / 50.0, 1.0) * 0.3
        multi_hop_words = {"compare", "difference", "relationship", "how", "why", "explain"}
        if any(w in query.lower() for w in multi_hop_words):
            complexity += 0.3
        if "?" in query:
            complexity += 0.1
        complexity += min(query.count(",") / 5.0, 0.3)
        return min(complexity, 1.0)

    @staticmethod
    def _estimate_confidence(answer: str) -> float:
        """Heuristic answer confidence estimation."""
        if not answer:
            return 0.0
        confidence = 0.5
        if len(answer.split()) > 5:
            confidence += 0.2
        hedging = {"maybe", "possibly", "uncertain", "not sure", "i think"}
        if any(h in answer.lower() for h in hedging):
            confidence -= 0.3
        return max(0.0, min(confidence, 1.0))


class PolicyNetwork(nn.Module):
    """Simple MLP policy head for action selection (used on top of LLM features)."""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 128, num_actions: int = 7):
        super().__init__()
        self.policy_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        self.value_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.policy_head(state_features)
        value = self.value_head(state_features).squeeze(-1)
        return logits, value

    def get_action(self, state_features: torch.Tensor, temperature: float = 1.0) -> Tuple[int, float]:
        logits, value = self.forward(state_features)
        probs = F.softmax(logits / temperature, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()
