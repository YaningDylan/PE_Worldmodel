"""Verify that HF repos in $HF_HOME are complete (every file present with the Hub's size).

Usage: $PE_PY serve/scripts/verify_download.py Qwen/Qwen3.5-9B [Qwen/Qwen3.6-27B ...]
Exit code 0 iff all repos are complete. Prints the local snapshot path and total bytes.
"""
import os
import sys

from huggingface_hub import HfApi

ok = True
api = HfApi()
for repo in sys.argv[1:]:
    info = api.model_info(repo, files_metadata=True)
    snap = os.path.join(os.environ["HF_HOME"], "hub", "models--" + repo.replace("/", "--"), "snapshots", info.sha)
    problems, total = [], 0
    for s in info.siblings:
        p = os.path.join(snap, s.rfilename)
        if not os.path.exists(p):
            problems.append(f"missing {s.rfilename}")
            continue
        sz = os.path.getsize(p)
        total += sz
        if s.size is not None and sz != s.size:
            problems.append(f"size {s.rfilename} {sz}!={s.size}")
    status = "COMPLETE" if not problems else "INCOMPLETE"
    ok &= not problems
    print(f"{status} {repo} sha={info.sha} files={len(info.siblings)} bytes={total} ({total/1e9:.2f} GB) path={snap}")
    for p in problems[:20]:
        print("   ", p)
sys.exit(0 if ok else 1)
