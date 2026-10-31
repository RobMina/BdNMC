"""The float32 switch (BDNMC_JAX_PRECISION): read at import, so each precision runs in its own process."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import bdnmc_jax
from bdnmc_jax import pipeline, shards

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "phase0" / "cards" / "ref" / "mb_pi0_m005.dat"


def run_block(tmp_path, precision):
    out = tmp_path / f"{precision}.npz"
    env = os.environ | {"BDNMC_JAX_PRECISION": precision}
    subprocess.run([sys.executable, str(ROOT / "validation" / "run_port.py"), str(CARD), "65536", "0", "1", str(out),
                    "--seed", "11"], env=env, check=True, capture_output=True)
    return shards.load(out)


def test_default_is_float64():
    assert bdnmc_jax.PRECISION == "float64"


def test_float32_run_is_float32_and_finite(tmp_path):
    res = run_block(tmp_path, "float32")
    c = res["candidates"]
    assert c["prob"].dtype == np.float32 and c["electron"].dtype == np.float32
    assert res["n_hits"] > 0 and np.isfinite(c["electron"]).all() and (c["prob"] >= 0).all()
    out = pipeline.finish(res)
    assert out["accepted"] > 0 and np.isfinite(out["se"]) and out["se"] > 0


def test_unknown_precision_is_refused(tmp_path):
    env = os.environ | {"BDNMC_JAX_PRECISION": "float16"}
    r = subprocess.run([sys.executable, "-c", "import bdnmc_jax"], env=env, capture_output=True, text=True,
                       cwd=ROOT / "src")
    assert r.returncode != 0 and "use float64 or float32" in r.stderr
