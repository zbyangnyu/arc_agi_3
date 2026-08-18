#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT=/root/autodl-tmp/lewm_pusht_tier1_5090_20260724
REPO_DIR="$RUN_ROOT/le-wm"
VENV_DIR="$RUN_ROOT/venv"
STABLEWM_DIR="$RUN_ROOT/stablewm"
RESULT_JSON="$RUN_ROOT/attempt7_result.json"
LOG_FILE="$RUN_ROOT/attempt7_run.log"
PIP_FREEZE="$RUN_ROOT/attempt7_pip_freeze.txt"
CURRENT_STAGE=initialization

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$LOG_FILE") 2>&1

write_failure() {
  rc=$?
  set +e
  if [[ ! -f "$RESULT_JSON" ]]; then
    /root/miniconda3/bin/python - "$RESULT_JSON" "$CURRENT_STAGE" "$rc" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": "lewm-pusht-tier1-5090-v1",
    "status": "error",
    "failed_stage": sys.argv[2],
    "exit_code": int(sys.argv[3]),
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
  fi
  exit "$rc"
}
trap write_failure ERR

CURRENT_STAGE=clone
if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone --depth 1 https://github.com/lucas-maes/le-wm.git "$REPO_DIR"
fi

CURRENT_STAGE=venv
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  /root/miniconda3/bin/python -m venv --system-site-packages "$VENV_DIR"
fi

CURRENT_STAGE=install
"$VENV_DIR/bin/python" -m pip install --no-cache-dir --upgrade pip
"$VENV_DIR/bin/python" -m pip install --no-cache-dir \
  'stable-worldmodel[train]' 'transformers>=4.50,<5' huggingface_hub imageio
"$VENV_DIR/bin/python" -m pip freeze > "$PIP_FREEZE"

CURRENT_STAGE=checkpoint_download
mkdir -p "$STABLEWM_DIR/hf_pusht"
HF_ENDPOINT=https://hf-mirror.com \
HF_HUB_DISABLE_XET=1 \
STABLEWM_HOME="$STABLEWM_DIR" \
"$VENV_DIR/bin/hf" download \
  quentinll/lewm-pusht \
  --local-dir "$STABLEWM_DIR/hf_pusht"

CURRENT_STAGE=checkpoint_convert_and_load
(
  cd "$REPO_DIR"
  export STABLEWM_HOME="$STABLEWM_DIR"
  export RESULT_JSON
  "$VENV_DIR/bin/python" - <<'PY'
import hashlib
import importlib.metadata
import json
import os
import pathlib
import subprocess
import time
import traceback
from datetime import datetime, timezone

import torch
import stable_pretraining as spt
import stable_worldmodel as swm

from jepa import JEPA
from module import ARPredictor, Embedder, MLP

result_path = pathlib.Path(os.environ["RESULT_JSON"])
cache = pathlib.Path(os.environ["STABLEWM_HOME"])
src = cache / "hf_pusht"
out = cache / "checkpoints" / "pusht" / "lewm_object.ckpt"


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def package_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def shape_summary(value):
    if isinstance(value, torch.Tensor):
        return {"kind": "tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (list, tuple)):
        return [shape_summary(x) for x in value]
    if isinstance(value, dict):
        return {str(k): shape_summary(v) for k, v in value.items()}
    return {"kind": type(value).__name__}


payload = {
    "schema_version": "lewm-pusht-tier1-5090-v1",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "status": "error",
}

try:
    cfg = json.loads((src / "config.json").read_text())
    encoder = spt.backbone.utils.vit_hf(
        cfg["encoder"]["size"],
        patch_size=cfg["encoder"]["patch_size"],
        image_size=cfg["encoder"]["image_size"],
        pretrained=False,
        use_mask_token=False,
    )

    def mlp(key):
        return MLP(
            input_dim=cfg[key]["input_dim"],
            output_dim=cfg[key]["output_dim"],
            hidden_dim=cfg[key]["hidden_dim"],
            norm_fn=torch.nn.BatchNorm1d,
        )

    model = JEPA(
        encoder=encoder,
        predictor=ARPredictor(
            **{
                key: value
                for key, value in cfg["predictor"].items()
                if not key.startswith("_")
            }
        ),
        action_encoder=Embedder(
            **{
                key: value
                for key, value in cfg["action_encoder"].items()
                if not key.startswith("_")
            }
        ),
        projector=mlp("projector"),
        pred_proj=mlp("pred_proj"),
    )
    state_dict = torch.load(
        src / "weights.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(state_dict, strict=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, out)

    torch.cuda.synchronize()
    load_start = time.perf_counter()
    loaded = swm.policy.AutoCostModel("pusht/lewm")
    loaded = loaded.to("cuda").eval()
    loaded.requires_grad_(False)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start

    forward = {"status": "not_attempted"}
    if hasattr(loaded, "encoder"):
        try:
            x = torch.randn(1, 3, 224, 224, device="cuda")
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                y = loaded.encoder(x)
            torch.cuda.synchronize()
            forward = {
                "status": "ok",
                "seconds": time.perf_counter() - start,
                "output": shape_summary(y),
            }
        except Exception as exc:
            forward = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    gpu_query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    repo_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    repo_status = subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    ).strip()

    payload.update(
        {
            "status": "ok",
            "repo": {
                "url": "https://github.com/lucas-maes/le-wm.git",
                "commit": repo_commit,
                "dirty": bool(repo_status),
            },
            "runtime": {
                "python": os.sys.version.split()[0],
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0),
                "nvidia_smi": gpu_query,
                "packages": {
                    name: package_version(name)
                    for name in [
                        "stable-worldmodel",
                        "stable-pretraining",
                        "huggingface-hub",
                        "transformers",
                        "torchvision",
                    ]
                },
            },
            "checkpoint": {
                "source_repo": "quentinll/lewm-pusht",
                "source_revision": (
                    src
                    / ".cache"
                    / "huggingface"
                    / "download"
                    / "weights.pt.metadata"
                ).read_text().splitlines()[0],
                "weights_path": str(src / "weights.pt"),
                "weights_bytes": (src / "weights.pt").stat().st_size,
                "weights_sha256": sha256(src / "weights.pt"),
                "config_sha256": sha256(src / "config.json"),
                "object_path": str(out),
                "object_bytes": out.stat().st_size,
                "object_sha256": sha256(out),
            },
            "model": {
                "parameter_count": sum(p.numel() for p in loaded.parameters()),
                "trainable_parameter_count_after_load": sum(
                    p.numel() for p in loaded.parameters() if p.requires_grad
                ),
                "training_mode_after_eval": loaded.training,
                "load_to_cuda_seconds": load_seconds,
                "encoder_forward": forward,
            },
            "conversion": {
                "config_top_level_keys": sorted(cfg),
                "predictor_config_keys": sorted(cfg["predictor"]),
                "action_encoder_config_keys": sorted(cfg["action_encoder"]),
                "hydra_metadata_keys_ignored": {
                    "predictor": sorted(
                        key
                        for key in cfg["predictor"]
                        if key.startswith("_")
                    ),
                    "action_encoder": sorted(
                        key
                        for key in cfg["action_encoder"]
                        if key.startswith("_")
                    ),
                },
                "state_dict_entries": len(state_dict),
                "state_dict_strict": True,
            },
            "scope": {
                "dataset_downloaded": False,
                "mpc_run": False,
            },
        }
    )
except Exception as exc:
    payload.update(
        {
            "status": "error",
            "failed_stage": "checkpoint_convert_and_load",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    )
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    raise

result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
)

CURRENT_STAGE=complete
echo "Tier 1 complete: $RESULT_JSON"
