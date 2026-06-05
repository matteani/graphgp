#!/usr/bin/env bash
set -euo pipefail

# Run from your activated local GraphGP/NIFTy environment.
# Example:
#   source ~/venvs/graphgp-dev/bin/activate
#   bash check_local_gpu_env.sh

export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.80}

python - <<'PY'
import jax
print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())

try:
    import graphgp
    print("graphgp: OK")
except Exception as e:
    print("graphgp import failed:", repr(e))
    raise

try:
    import nifty.re as jft
    print("nifty.re: OK")
except Exception as e1:
    try:
        import nifty8.re as jft
        print("nifty8.re: OK")
    except Exception as e2:
        print("NIFTy import failed")
        print("nifty.re:", repr(e1))
        print("nifty8.re:", repr(e2))
        raise
PY
