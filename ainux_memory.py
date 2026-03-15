"""
AInux Memory Module
Hierarchical memory with FAISS vector store, sentence-transformer embeddings,
and time-decay scoring. Persists across sessions via disk.
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class MemoryItem:
    id: str
    text: str                      # human-readable description
    command: str                   # the actual shell command stored
    intent: str                    # classified intent label
    outcome: str                   # "success" | "failure"
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    layer: str = "short"           # "short" | "mid" | "long"
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryItem":
        return cls(**d)


# ---------------------------------------------------------------------------
# Memory store
# ---------------------------------------------------------------------------

class AInuxMemory:
    """
    Three-layer memory system:
      - short : current session embeddings (in-memory only)
      - mid   : this session's state transitions (persisted at end of session)
      - long  : stable preferences and repeated workflows (always persisted)

    Vector index: FAISS IndexFlatIP (inner-product = cosine after L2-norm).
    Embeddings: sentence-transformers all-MiniLM-L6-v2  (384-dim, 80 MB).
    Decay: score(m, t) = sim(q, m) * exp(-λ*(t-t_m)) * (1 + log(1 + f_m))
    """

    EMBED_MODEL = "all-MiniLM-L6-v2"
    DIM = 384
    DECAY_LAMBDA = 1e-5          # ~11.5 days half-life
    PERSIST_DIR = Path.home() / ".ainux" / "memory"

    def __init__(self, persist: bool = True):
        self.persist = persist
        self.PERSIST_DIR.mkdir(parents=True, exist_ok=True)

        self._encoder = SentenceTransformer(self.EMBED_MODEL)

        # FAISS index (inner product; we L2-normalise so IP == cosine)
        self._index = faiss.IndexFlatIP(self.DIM)

        # Parallel list of MemoryItems (same order as FAISS vectors)
        self._items: List[MemoryItem] = []

        if persist:
            self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(
        self,
        text: str,
        command: str,
        intent: str,
        outcome: str,
        metadata: Optional[Dict] = None,
        layer: str = "short",
    ) -> MemoryItem:
        """Encode and store a memory item. Returns the stored item."""
        item = MemoryItem(
            id=str(uuid.uuid4()),
            text=text,
            command=command,
            intent=intent,
            outcome=outcome,
            layer=layer,
            metadata=metadata or {},
        )
        vec = self._encode(text)
        self._index.add(vec)
        self._items.append(item)

        if self.persist and layer in ("mid", "long"):
            self._save()

        return item

    def retrieve(self, query: str, k: int = 5, layer: Optional[str] = None) -> List[Tuple[MemoryItem, float]]:
        """
        Return top-k items ranked by decayed cosine similarity.
        Optionally filter by memory layer.
        """
        if self._index.ntotal == 0:
            return []

        q_vec = self._encode(query)
        # Over-fetch then re-rank with decay
        fetch_k = min(self._index.ntotal, k * 4)
        sims, indices = self._index.search(q_vec, fetch_k)

        now = time.time()
        scored: List[Tuple[MemoryItem, float]] = []

        for sim, idx in zip(sims[0], indices[0]):
            if idx < 0 or idx >= len(self._items):
                continue
            item = self._items[idx]
            if layer and item.layer != layer:
                continue
            decayed = self._decay_score(float(sim), item, now)
            scored.append((item, decayed))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def consolidate(self) -> None:
        """
        Promote frequently accessed short/mid memories to long-term.
        Uses access_count threshold: items accessed >= 3 times become 'long'.
        Also prunes short-term memories older than 1 hour.
        """
        now = time.time()
        promoted = 0
        pruned = 0

        for item in self._items:
            if item.layer == "short":
                age = now - item.timestamp
                if age > 3600:          # older than 1 hour
                    item.layer = "pruned"
                    pruned += 1
                elif item.access_count >= 3:
                    item.layer = "long"
                    promoted += 1
            elif item.layer == "mid" and item.access_count >= 3:
                item.layer = "long"
                promoted += 1

        if self.persist:
            self._save()

        return {"promoted": promoted, "pruned": pruned}

    def recall_last(self, intent: str) -> Optional[MemoryItem]:
        """Return the most recent successful memory matching an intent."""
        candidates = [
            i for i in reversed(self._items)
            if i.intent == intent and i.outcome == "success" and i.layer != "pruned"
        ]
        if candidates:
            candidates[0].access_count += 1
            return candidates[0]
        return None

    def stats(self) -> Dict:
        active = [i for i in self._items if i.layer != "pruned"]
        return {
            "total": len(active),
            "short": sum(1 for i in active if i.layer == "short"),
            "mid":   sum(1 for i in active if i.layer == "mid"),
            "long":  sum(1 for i in active if i.layer == "long"),
            "index_size": self._index.ntotal,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, text: str) -> np.ndarray:
        """Return L2-normalised embedding as (1, DIM) float32 array."""
        vec = self._encoder.encode([text], normalize_embeddings=True)
        return vec.astype("float32")

    def _decay_score(self, sim: float, item: MemoryItem, now: float) -> float:
        """
        score(m, t) = sim(q, m) * exp(-λ*(t - t_m)) * (1 + log(1 + f_m))
        Matches Equation 9 in the paper.
        """
        age = now - item.timestamp
        decay = math.exp(-self.DECAY_LAMBDA * age)
        freq_boost = 1.0 + math.log1p(item.access_count)
        return sim * decay * freq_boost

    def _items_path(self) -> Path:
        return self.PERSIST_DIR / "items.json"

    def _save(self) -> None:
        saveable = [i.to_dict() for i in self._items if i.layer != "pruned"]
        with open(self._items_path(), "w") as f:
            json.dump(saveable, f, indent=2)

        # Rebuild FAISS index from saved items so it stays in sync
        if saveable:
            texts = [d["text"] for d in saveable]
            vecs = self._encoder.encode(texts, normalize_embeddings=True).astype("float32")
            self._index = faiss.IndexFlatIP(self.DIM)
            self._index.add(vecs)

    def _load(self) -> None:
        path = self._items_path()
        if not path.exists():
            return
        with open(path) as f:
            data = json.load(f)

        self._items = [MemoryItem.from_dict(d) for d in data]

        # Rebuild FAISS index
        if self._items:
            texts = [i.text for i in self._items]
            vecs = self._encoder.encode(texts, normalize_embeddings=True).astype("float32")
            self._index = faiss.IndexFlatIP(self.DIM)
            self._index.add(vecs)
