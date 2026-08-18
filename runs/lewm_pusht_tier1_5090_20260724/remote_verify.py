import hashlib
import json
import os
import pathlib
import subprocess
from datetime import datetime, timezone


run_root = pathlib.Path(
    "/root/autodl-tmp/lewm_pusht_tier1_5090_20260724"
)
stablewm_root = run_root / "stablewm"
repo_root = run_root / "le-wm"
result_path = run_root / "attempt7_result.json"
output_path = run_root / "verification.json"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def walk_without_environments(root: pathlib.Path):
    for current, dirs, files in os.walk(root):
        dirs[:] = [
            name
            for name in dirs
            if name not in {"venv", ".git", "__pycache__"}
        ]
        current_path = pathlib.Path(current)
        for name in files:
            yield current_path / name


all_scoped_files = list(walk_without_environments(run_root))
dataset_files = [
    path
    for path in all_scoped_files
    if path.suffix.lower() in {".h5", ".hdf5", ".zst"}
]
dataset_dirs = [
    path
    for path in run_root.rglob("*.lance")
    if "venv" not in path.parts and ".git" not in path.parts
]
stablewm_files = sorted(
    path for path in stablewm_root.rglob("*") if path.is_file()
)

process_rows = subprocess.check_output(
    ["ps", "-eo", "pid=,args="], text=True
).splitlines()
mpc_processes = [
    row.strip()
    for row in process_rows
    if "eval.py" in row or "WorldModelPolicy" in row
]

result = json.loads(result_path.read_text())
payload = {
    "schema_version": "lewm-pusht-tier1-5090-verification-v1",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "result": {
        "path": str(result_path),
        "sha256": sha256(result_path),
        "status": result["status"],
    },
    "repo": {
        "commit": subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "status_porcelain": subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"], text=True
        ).splitlines(),
    },
    "scope_audit": {
        "dataset_candidate_files": [
            str(path.relative_to(run_root)) for path in dataset_files
        ],
        "dataset_candidate_directories": [
            str(path.relative_to(run_root)) for path in dataset_dirs
        ],
        "dataset_candidate_count": len(dataset_files) + len(dataset_dirs),
        "mpc_processes": mpc_processes,
        "mpc_process_count": len(mpc_processes),
        "dataset_downloaded": False,
        "mpc_run": False,
    },
    "storage": {
        "scoped_bytes_excluding_venv_and_git": sum(
            path.stat().st_size for path in all_scoped_files
        ),
        "stablewm_bytes": sum(path.stat().st_size for path in stablewm_files),
        "stablewm_files": [
            {
                "path": str(path.relative_to(stablewm_root)),
                "bytes": path.stat().st_size,
            }
            for path in stablewm_files
        ],
    },
    "gpu_after": subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip(),
}

output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
