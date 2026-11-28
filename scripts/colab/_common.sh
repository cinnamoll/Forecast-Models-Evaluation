# Sourced by the Colab wrappers (02_tune, 03_run, 04_xai_nf, 05_xai_gluonts).
#
# Two things the wrappers kept getting wrong:
#
# 1. /content/env.sh exports REPO_DIR from the clone it was written for.  A second
#    clone (…-1) then ran the roster against the *first* checkout, with a traceback
#    naming a path the user never typed.  The script knows where it lives, so that
#    wins, and a differing REPO_DIR is reported instead of silently used.
# 2. The roster's launcher needs pandas (it imports utils.protocol) but env.sh puts
#    the conda *base* first on PATH, and the base env has no pandas -- and in a
#    gluonts-only session there is no neuralforecast env to borrow either.  So scan
#    every env of CONDA_HOME and take the first python that imports pandas, newest
#    interpreter first: the driver also imports repo modules that assume 3.8+ syntax.
#
# Override with PYTHON=/path/to/python when a specific interpreter is wanted.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

if [ -n "${REPO_DIR:-}" ] && [ "$REPO_DIR" != "$ROOT" ]; then
  echo "NOTE: REPO_DIR=$REPO_DIR is another checkout; using the repo this script lives in: $ROOT" >&2
fi
REPO_DIR="$ROOT"
export REPO_DIR

_conda_home="${CONDA_HOME:-/content/miniconda3}"

# python >= 3.8 that can import pandas, else any python with pandas (gluonts' 3.7).
_pick_python() {
  local minimum="$1" candidate
  for candidate in "$_conda_home"/envs/*/bin/python "$_conda_home/bin/python" python3 python; do
    [ -x "$candidate" ] || continue
    if "$candidate" - "$minimum" >/dev/null 2>&1 <<'PY'
import sys
minimum = tuple(int(part) for part in sys.argv[1].split(".")) if sys.argv[1] != "any" else (0,)
import pandas  # noqa: F401
raise SystemExit(0 if sys.version_info[:2] >= minimum else 1)
PY
    then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

if [ -z "${PYTHON:-}" ]; then
  if PYTHON="$(_pick_python 3.8)"; then
    :
  elif PYTHON="$(_pick_python any)"; then
    echo "NOTE: no python 3.8+ with pandas found; using $PYTHON (gluonts env is 3.7)" >&2
  else
    echo "FAIL: no python with pandas under $_conda_home -- run scripts/colab/01_setup.sh first" >&2
    exit 2
  fi
fi
export PYTHON
