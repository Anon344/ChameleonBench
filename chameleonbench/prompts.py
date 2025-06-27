from dataclasses import dataclass
from pathlib import Path
import json, random
from typing import List, Optional

@dataclass
class PromptPair:
    id: str
    category: str
    eval_prompt: str
    deploy_prompt: str

def load_pairs(path: str | Path,
               n: Optional[int] = None,
               seed: int = 42) -> List[PromptPair]:
    path = Path(path)
    with path.open(encoding="utf-8") as fp:
        raw = json.load(fp)
    rng = random.Random(seed)
    rng.shuffle(raw)
    if n:
        raw = raw[:n]
    return [PromptPair(**row) for row in raw]
