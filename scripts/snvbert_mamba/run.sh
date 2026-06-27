#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
TRAIN_PACK_DIR="${TRAIN_PACK_DIR:?Set TRAIN_PACK_DIR}"
BACKEND_IDENTITY="${BACKEND_IDENTITY:?Set BACKEND_IDENTITY}"
OUTPUT="${OUTPUT:?Set OUTPUT}"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29683}"

PYTHON_BIN="$(command -v python || true)"
TORCHRUN_BIN="$(command -v torchrun || true)"
[[ -x "${PYTHON_BIN}" ]] || { echo "[ERROR] python unavailable" >&2; exit 2; }
[[ -x "${TORCHRUN_BIN}" ]] || { echo "[ERROR] torchrun unavailable" >&2; exit 2; }
[[ -d "${ROOT_DIR}" ]] || { echo "[ERROR] source root missing: ${ROOT_DIR}" >&2; exit 2; }
[[ -d "${TRAIN_PACK_DIR}" ]] || { echo "[ERROR] training pack missing: ${TRAIN_PACK_DIR}" >&2; exit 2; }
[[ ! -e "${OUTPUT}" ]] || { echo "[ERROR] refusing to overwrite ${OUTPUT}" >&2; exit 2; }

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

"${PYTHON_BIN}" -m py_compile \
  "${ROOT_DIR}/src/train_snvbert_mamba.py" \
  "${ROOT_DIR}/src/snvbert_mamba/model.py" \
  "${ROOT_DIR}/src/snvbert_mamba/layers.py" \
  "${ROOT_DIR}/src/snvbert_mamba/loss.py"
"${PYTHON_BIN}" "${ROOT_DIR}/src/train_snvbert_mamba.py" --help >/dev/null

if [[ "${PREFLIGHT_ONLY:-false}" == true ]]; then
  echo "[PASS] standalone source, runtime and training-data paths verified"
  exit 0
fi

mkdir -m700 -p "${OUTPUT}/logs"
set +e
"${TORCHRUN_BIN}" \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${ROOT_DIR}/src/train_snvbert_mamba.py" \
  --train-pack-dir "${TRAIN_PACK_DIR}" \
  --output "${OUTPUT}" \
  --backend metax_mamba2 \
  --backend-identity "${BACKEND_IDENTITY}" \
  --geometry none \
  --global-batch 128 \
  --micro-batch "${MICRO_BATCH:-1}" \
  --maximum-updates "${MAXIMUM_UPDATES:-50}" \
  --warmup-updates "${WARMUP_UPDATES:-10}" \
  >"${OUTPUT}/logs/stdout.log" \
  2>"${OUTPUT}/logs/stderr.log"
status=$?
set -e

printf '%s\n' "${status}" >"${OUTPUT}/RUNNER_EXIT"
if [[ "${status}" -ne 0 ]]; then
  tail -n 120 "${OUTPUT}/logs/stderr.log" >&2 || true
  exit "${status}"
fi
echo "[PASS] standalone training completed: ${OUTPUT}"
