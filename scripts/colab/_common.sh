# Sourced by the Colab wrappers (02_tune, 03_run, 04_xai_nf, 05_xai_gluonts).
#
# Two things the wrappers kept getting wrong:
#
# 1. /content/env.sh exports REPO_DIR from the clone it was written for.  A second
#    clone (…-1) then ran the roster against the *first* checkout, with a traceback
#    naming a path the user never typed.  The script knows where it lives, so that
#    wins, and a differing REPO_DIR is reported instead of silently used.
# 2. The roster's launcher needs pandas (it imports utils.protocol) but env.sh puts
#    the conda *base* first on PATH, and the base env has no pandas.  Pick a python
#    that can actually import it: the neuralforecast env, then base, then PATH.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

if [ -n "${REPO_DIR:-}" ] && [ "$REPO_DIR" != "$ROOT" ]; then
  echo "NOTE: REPO_DIR=$REPO_DIR is another checkout; using the repo this script lives in: $ROOT" >&2
fi
REPO_DIR="$ROOT"
export REPO_DIR

if [ -z "${PYTHON:-}" ]; then
  for candidate in "${CONDA_HOME:-/content/miniconda3}/envs/neuralforecast/bin/python" \
                   "${CONDA_HOME:-/content/miniconda3}/bin/python" \
                   python3 python; do
    if [ -x "$candidate" ] && "$candidate" -c "import pandas" >/dev/null 2>&1; then
      PYTHON="$candidate"
      break
    fi
  done
  PYTHON="${PYTHON:-python}"
fi
export PYTHON
