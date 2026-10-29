"""Save and load pass-1 shards (`pipeline.candidates`) as .npz, so disjoint block ranges can run in separate
processes and be merged afterwards."""
import json
from pathlib import Path

import numpy as np

from .pipeline import CANDIDATE_KEYS


def save(res, path):
    meta = {k: v for k, v in res.items() if k != "candidates"}
    np.savez(path, meta=json.dumps(meta), **{f"c_{k}": res["candidates"][k] for k in CANDIDATE_KEYS
                                             if k in res["candidates"]})


def load(path):
    """Shards written before a key was added to CANDIDATE_KEYS load without it."""
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        return meta | {"candidates": {k: z[f"c_{k}"] for k in CANDIDATE_KEYS if f"c_{k}" in z.files}}


def load_all(paths):
    return [load(p) for p in sorted(Path(p) for p in paths)]
