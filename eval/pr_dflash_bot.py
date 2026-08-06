#!/usr/bin/env python3
"""sparkinfer DFlash PR auto-evaluator.

Sibling of pr_eval_bot.py for PRs that touch the DFlash speculative-decode path.
Scores same-box PR DFlash tok/s vs origin/main DFlash tok/s, applies eval-dflash:*
tiers, picks dflash-merge-first, and optionally auto-merges (SPARKINFER_AUTOMERGE=1).

  python eval/pr_dflash_bot.py --instance 46074104
  python eval/pr_dflash_bot.py --only-prs 636 --reeval

Never rents a GPU. Shares the pinned box with the AR bot via flock in the cron wrapper.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ssh_box import ssh_box_enabled, ssh_box_endpoint  # noqa: E402

# Reuse shared helpers from the AR bot (labels, greenlight, denylist, …).
import pr_eval_bot as arb  # noqa: E402

SPEEDUP_LABELS = {"XL", "L", "M", "S", "XS"}
SIG = 0.02
REGRESS_TOL = 0.98
BUCKETS = [(0.18, "XL"), (0.10, "L"), (0.06, "M"), (0.035, "S"), (SIG, "XS")]

EVAL_PREFIX = "eval-dflash:"
DFLASH_MERGE_FIRST = "dflash-merge-first"
DFLASH_NEEDS_REBASE = "dflash-needs-rebase"
# Bumped when the scoring/guard logic changes materially (e.g. adding the Qwen3.5/3.6
# no-regression guard) — old markers from before the bump deliberately DON'T match, so a PR
# whose head commit hasn't moved since a pre-guard eval is treated as never-evaluated and gets
# a fresh, guarded run instead of keeping its stale (unguarded) label/score forever.
EVAL_SCHEMA_VERSION = "v2-qwenguard"
MARKER_RE = re.compile(
    r"<!-- sparkinfer-dflash-eval:" + re.escape(EVAL_SCHEMA_VERSION) + r":([0-9a-f]+)(?:\s+(\{.*?\}))? -->",
    re.DOTALL,
)
DFLASH_PATH_RE = re.compile(r"(?:^|/)(?:dflash|qwen3_gguf_dflash_|dflash_accuracy\.sh)", re.I)

DEFAULT_GGUF = os.environ.get(
    "DFLASH_GGUF", "/workspace/models36/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
)
DEFAULT_DRAFT = os.environ.get(
    "DFLASH_DRAFT", "/workspace/models_dflash/Qwen3.6-35B-A3B-DFlash"
)
DEFAULT_MODELS_DIR = os.environ.get("DFLASH_MODELS_DIR", "/workspace/models36")
REMOTE_REPO = os.environ.get("DFLASH_REMOTE_REPO", "/root/sparkinfer")
BENCH_TOKENS = int(os.environ.get("DFLASH_BENCH_TOKENS", "128"))

# Qwen3.5 (Qwythos) + Qwen3.6 no-regression guard — since #667-era policy, DFlash is the only
# thing that gets a score; it's only valid if the *same PR build* doesn't regress Qwen3.5/3.6
# decode or prefill vs same-box origin/main. Qwen3.6 reuses the DFlash GGUF (one copy, no extra
# download); Qwen3.5 uses the standard Qwythos path already used by the AR/bidir bot.
Q36_GUARD_MODEL_FILE = os.path.basename(DEFAULT_GGUF)
Q36_GUARD_MODELS_DIR = os.path.dirname(DEFAULT_GGUF) or DEFAULT_MODELS_DIR
Q36_GUARD_MODEL_REPO = os.environ.get("PRIMARY36_MODEL_REPO", "unsloth/Qwen3.6-35B-A3B-GGUF")
Q36_GUARD_TOK_REPO = os.environ.get("PRIMARY36_TOK_REPO", "Qwen/Qwen3.6-35B-A3B")
Q35_GUARD_MODELS_DIR = os.environ.get("QWYTHOS_MODELS_DIR", "/workspace/models35")
GUARD_CTX_LABEL = {0: "128", 512: "512", 4096: "4k", 16384: "16k", 32768: "32k",
                   65536: "64k", 131072: "128k"}

AUTO_MERGE = os.environ.get("SPARKINFER_AUTOMERGE", "0") == "1"
AUTOMERGE_BLOCK = {
    "copycat", "copycat-warn", "flagged:gaming", "penalty", "needs-benchmark",
    DFLASH_NEEDS_REBASE, arb.REEVALUATE_LABEL, arb.HOLD_LABEL, *arb.REGRESSION_LABELS,
}

SCORES_FILE = os.path.expanduser(
    os.environ.get("DFLASH_SCORES_FILE", "~/.sparkinfer_dflash_scores.json")
)


def _load_scores():
    try:
        return json.load(open(SCORES_FILE))
    except Exception:
        return {}


def _save_scores(data):
    try:
        with open(SCORES_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f">> dflash scores save skipped: {e}")


def tier_from_gain(pr_tps: float, main_tps: float):
    """Return (label, delta_pct, pass_ok, reason)."""
    if main_tps <= 0:
        return "REJECT", 0.0, False, "main DFlash baseline is 0"
    if pr_tps < REGRESS_TOL * main_tps:
        pct = 100.0 * (pr_tps - main_tps) / main_tps
        return "REJECT", round(pct, 1), False, (
            f"DFlash regression: {pr_tps:.2f} < {100 * REGRESS_TOL:.0f}% of main {main_tps:.2f}"
        )
    g = (pr_tps - main_tps) / main_tps
    pct = round(100.0 * g, 1)
    if g < SIG:
        return "none", pct, True, "within significance gate — not a verified DFlash improvement"
    for thr, name in BUCKETS:
        if g >= thr:
            return name, pct, True, "ok"
    return "none", pct, True, "ok"


def is_dflash_pr(repo, num) -> bool:
    files = json.loads(
        arb.gh(["pr", "view", str(num), "-R", repo, "--json", "files"]).stdout or "{}"
    ).get("files", [])
    for f in files:
        path = f.get("path") or ""
        if DFLASH_PATH_RE.search(path):
            return True
    return False


def dflash_evaluated_commits(repo, num):
    r = arb.gh(["pr", "view", str(num), "-R", repo, "--json", "comments"])
    done = set()
    for c in json.loads(r.stdout or "{}").get("comments", []):
        m = MARKER_RE.search(c.get("body") or "")
        if m and "sparkinfer dflash auto-eval" in (c.get("body") or ""):
            done.add(m.group(1))
    return done


def strip_dflash_eval_labels(repo, num):
    for lab in list(arb.labels_on(repo, num)):
        if lab.startswith(EVAL_PREFIX):
            arb.remove_label(repo, num, lab)


def resolve_ssh(instance_id: int):
    """Return (host, port) for the pinned box."""
    if ssh_box_enabled():
        ep = ssh_box_endpoint()
        if not ep:
            raise RuntimeError("EVAL_TRANSPORT=ssh but EVAL_SSH_HOST unset")
        return ep
    key = os.environ.get("SSH_KEY", os.path.expanduser("~/.ssh/speedy"))
    os.environ.setdefault("SSH_KEY", key)
    iid = arb.current_instance(instance_id) or instance_id
    raw = subprocess.run(
        ["vastai", "show", "instance", str(iid), "--raw"],
        capture_output=True, text=True, timeout=60,
    )
    if raw.returncode != 0 or not (raw.stdout or "").strip():
        raise RuntimeError(f"vastai show instance {iid} failed: {(raw.stderr or '')[:200]}")
    info = json.loads(raw.stdout)
    ip = (info.get("public_ipaddr") or "").strip()
    ports = info.get("ports") or {}
    m = ports.get("22/tcp") or [{}]
    port = int((m[0] or {}).get("HostPort") or 0)
    if info.get("actual_status") != "running" or not ip or not port:
        raise RuntimeError(
            f"pinned instance {iid} not SSH-ready (status={info.get('actual_status')})"
        )
    return ip, port


def ssh_run(host, port, cmd, timeout=7200):
    key = os.environ.get("SSH_KEY", os.path.expanduser("~/.ssh/speedy"))
    return subprocess.run(
        [
            "ssh", "-i", key,
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "BatchMode=yes",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=40",
            "-p", str(port), f"root@{host}", cmd,
        ],
        capture_output=True, text=True, timeout=timeout,
    )


def _remote_script(ref: str, do_accuracy: bool, prompt_ids: str | None, n_tokens: int) -> str:
    """Bash run on the eval box: checkout ref, build, optional accuracy, bench."""
    gguf = shlex.quote(DEFAULT_GGUF)
    draft = shlex.quote(DEFAULT_DRAFT)
    models = shlex.quote(DEFAULT_MODELS_DIR)
    repo = shlex.quote(REMOTE_REPO)
    ref_q = shlex.quote(ref)
    hf = shlex.quote(os.environ.get("HF_TOKEN", ""))
    ids_export = ""
    if prompt_ids:
        ids_export = f"PROMPT_IDS={shlex.quote(prompt_ids)}\n"
    acc = "1" if do_accuracy else "0"
    q36_file = shlex.quote(Q36_GUARD_MODEL_FILE)
    q36_dir = shlex.quote(Q36_GUARD_MODELS_DIR)
    q36_repo = shlex.quote(Q36_GUARD_MODEL_REPO)
    q36_tok = shlex.quote(Q36_GUARD_TOK_REPO)
    q35_dir = shlex.quote(Q35_GUARD_MODELS_DIR)
    return f"""
set -euo pipefail
export PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH
export CUDA_HOME=${{CUDA_HOME:-/usr/local/cuda-13.0}}
export HF_TOKEN={hf}
export HF_HUB_DISABLE_XET=1
REPO={repo}
GGUF={gguf}
DRAFT={draft}
MODELS_DIR={models}
NTOK={n_tokens}
DO_ACC={acc}
Q36_GUARD_MODEL_FILE={q36_file}
Q36_GUARD_MODELS_DIR={q36_dir}
Q36_GUARD_MODEL_REPO={q36_repo}
Q36_GUARD_TOK_REPO={q36_tok}
Q35_GUARD_MODELS_DIR={q35_dir}
{ids_export}
cd "$REPO"
git remote set-url origin https://github.com/gittensor-ai-lab/sparkinfer.git 2>/dev/null || true
git fetch -q origin {ref_q}
git reset -q --hard
git clean -qfd
git checkout -qf FETCH_HEAD
HEAD=$(git rev-parse --short HEAD)
echo "REMOTE_HEAD $HEAD"

# Ensure draft weights exist
if [ ! -f "$DRAFT/model.safetensors" ] && [ ! -f "$DRAFT/model.safetensors.index.json" ]; then
  mkdir -p "$(dirname "$DRAFT")"
  hf download z-lab/Qwen3.6-35B-A3B-DFlash --local-dir "$DRAFT"
fi
test -f "$GGUF" || {{ echo "FAIL missing GGUF $GGUF"; exit 1; }}

# Build dflash tools + qwen3_gguf_bench (incremental) — the latter drives the Qwen3.5/3.6 guard below.
mkdir -p build
if [ ! -f build/CMakeCache.txt ]; then
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/tmp/dflash_cmake.log 2>&1
fi
cmake --build build --target qwen3_gguf_dflash_check qwen3_gguf_dflash_bench qwen3_gguf_bench -j"$(nproc)" >/tmp/dflash_build.log 2>&1
test -x build/runtime/qwen3_gguf_dflash_bench
test -x build/runtime/qwen3_gguf_bench

if [ "$DO_ACC" = "1" ]; then
  export MODELS_DIR
  bash bench/scripts/dflash_accuracy.sh "$GGUF" "$DRAFT" | tee /tmp/dflash_check_out.txt
  grep -q "^VERDICT PASS" /tmp/dflash_check_out.txt
  if [ -z "${{PROMPT_IDS:-}}" ] && [ -f /tmp/dflash_eval_ids.txt ]; then
    PROMPT_IDS=$(cat /tmp/dflash_eval_ids.txt)
  fi
  echo "PROMPT_IDS $PROMPT_IDS"
  SPEC=$(grep '^METRIC SPEC_AGREE' /tmp/dflash_check_out.txt | tail -1 || true)
  echo "$SPEC"
else
  PROMPT_IDS="${{PROMPT_IDS:-}}"
  if [ -z "$PROMPT_IDS" ] && [ -f /tmp/dflash_eval_ids.txt ]; then
    PROMPT_IDS=$(cat /tmp/dflash_eval_ids.txt)
  fi
  echo "PROMPT_IDS $PROMPT_IDS"
fi

OUT=$(build/runtime/qwen3_gguf_dflash_bench "$GGUF" "$DRAFT" "$NTOK" $PROMPT_IDS | tee /tmp/dflash_bench_out.txt)
echo "$OUT"
AR=$(echo "$OUT" | grep '^METRIC AR_TPS' | awk '{{print $3}}' | tail -1)
DF=$(echo "$OUT" | grep '^METRIC DFLASH_TPS' | awk '{{print $3}}' | tail -1)
TAU=$(echo "$OUT" | grep '^METRIC MEAN_ACCEPT' | awk '{{print $3}}' | tail -1)
echo "RESULT_AR_TPS $AR"
echo "RESULT_DFLASH_TPS $DF"
echo "RESULT_MEAN_ACCEPT $TAU"

# --- Qwen3.5 / Qwen3.6 no-regression guard (decode + prefill, same build as above) ---
source bench/scripts/_common.sh
source bench/scripts/_eval_speed.sh
# Pin QWYTHOS_MODELS_DIR before sourcing _qwythos.sh: its own default derives from the ambient
# $MODELS_DIR, which the DFlash steps above already repointed at /workspace/models36 — falling
# through to that default here would silently resolve to .../models3635.
export QWYTHOS_MODELS_DIR="$Q35_GUARD_MODELS_DIR"
source bench/scripts/_qwythos.sh
SI_BIN="$PWD/build/runtime"; SI_LD=""
gclks=()

Q35_FILE="$(qwythos_quant_file)"
export MODELS_DIR="$QWYTHOS_MODELS_DIR" MODEL_REPO="$QWYTHOS_REPO" MODEL_FILE="$Q35_FILE" TOK_REPO="$QWYTHOS_TOK_REPO"
export MODEL_SHA256="$(qwythos_sha_var)"
( ensure_model && ensure_tokenizer ) || echo "WARN: qwen3.5 guard model setup failed" >&2
Q35_GGUF="$QWYTHOS_MODELS_DIR/$Q35_FILE"

export MODELS_DIR="$Q36_GUARD_MODELS_DIR" MODEL_REPO="$Q36_GUARD_MODEL_REPO" MODEL_FILE="$Q36_GUARD_MODEL_FILE" TOK_REPO="$Q36_GUARD_TOK_REPO"
export MODEL_SHA256="${{QWEN36_MODEL_SHA256:-}}"
( ensure_model && ensure_tokenizer ) || echo "WARN: qwen3.6 guard model setup failed" >&2
Q36_GGUF="$Q36_GUARD_MODELS_DIR/$Q36_GUARD_MODEL_FILE"

echo "GUARD_START"
if bench_sweep_run "$Q36_GGUF" 128 0 1 512 1 4096 1 16384 1 32768 1; then
  for ctx in 0 512 4096 16384 32768; do
    echo "GUARD36 $ctx $(_bench_sweep_get $ctx decode_tps) $(_bench_sweep_get $ctx prefill_pp)"
  done
else
  echo "GUARD36_FAILED"
fi
if bench_sweep_run "$Q35_GGUF" 128 0 1 4096 1 32768 1 65536 1 131072 1; then
  for ctx in 0 4096 32768 65536 131072; do
    echo "GUARD35 $ctx $(_bench_sweep_get $ctx decode_tps) $(_bench_sweep_get $ctx prefill_pp)"
  done
else
  echo "GUARD35_FAILED"
fi
echo "GUARD_END"
"""


def _parse_remote(stdout: str) -> dict:
    out = {}
    guard36, guard35 = {}, {}
    for line in (stdout or "").splitlines():
        if line.startswith("RESULT_AR_TPS "):
            out["ar_tps"] = float(line.split()[1])
        elif line.startswith("RESULT_DFLASH_TPS "):
            out["dflash_tps"] = float(line.split()[1])
        elif line.startswith("RESULT_MEAN_ACCEPT "):
            out["mean_accept"] = float(line.split()[1])
        elif line.startswith("PROMPT_IDS "):
            out["prompt_ids"] = line[len("PROMPT_IDS "):].strip()
        elif line.startswith("REMOTE_HEAD "):
            out["head"] = line.split()[1]
        elif line.startswith("METRIC SPEC_AGREE"):
            out["spec_agree"] = line.strip()
        elif line.startswith("GUARD36 "):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    guard36[int(parts[1])] = {"decode": float(parts[2]), "prefill": float(parts[3])}
                except ValueError:
                    pass
        elif line.startswith("GUARD35 "):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    guard35[int(parts[1])] = {"decode": float(parts[2]), "prefill": float(parts[3])}
                except ValueError:
                    pass
        elif line.strip() == "GUARD36_FAILED":
            out["guard36_failed"] = True
        elif line.strip() == "GUARD35_FAILED":
            out["guard35_failed"] = True
    out["guard36"] = guard36
    out["guard35"] = guard35
    return out


def check_qwen_guard(pr: dict, main: dict, tol: float = REGRESS_TOL):
    """No-regression check: PR vs same-box main, Qwen3.5 + Qwen3.6, decode + prefill, every
    measured context. Returns (ok, [human-readable regression/failure strings])."""
    problems = []
    if pr.get("guard36_failed") or main.get("guard36_failed") or not pr.get("guard36") or not main.get("guard36"):
        problems.append("qwen3.6 guard measurement unavailable")
    if pr.get("guard35_failed") or main.get("guard35_failed") or not pr.get("guard35") or not main.get("guard35"):
        problems.append("qwen3.5 guard measurement unavailable")
    # Iterate over MAIN's contexts (the reference/expected set), not the PR's — a PR build that
    # crashes partway through its sweep and never reports a context must not make that context
    # silently uncheckable. Fail closed: a real main baseline (base > 0) with a missing or zero
    # PR measurement (cur <= 0) is flagged as a regression, never skipped.
    for model_name, pr_ctxs, main_ctxs in (
        ("qwen3.6", pr.get("guard36") or {}, main.get("guard36") or {}),
        ("qwen3.5", pr.get("guard35") or {}, main.get("guard35") or {}),
    ):
        for ctx, main_vals in main_ctxs.items():
            label = GUARD_CTX_LABEL.get(ctx, str(ctx))
            pr_vals = pr_ctxs.get(ctx) or {}
            for metric in ("decode", "prefill"):
                base = main_vals.get(metric, 0)
                if base <= 0:
                    continue  # main itself has no baseline for this metric/ctx — not comparable
                cur = pr_vals.get(metric, 0)
                if cur <= 0:
                    problems.append(
                        f"{model_name} {metric}@{label}: PR measurement missing/zero "
                        f"(main {base:.1f}) — treated as regression"
                    )
                    continue
                if cur < base * tol:
                    pct = 100.0 * (cur - base) / base
                    problems.append(
                        f"{model_name} {metric}@{label}: {cur:.1f} < {100 * tol:.0f}% of main "
                        f"{base:.1f} ({pct:+.1f}%)"
                    )
    return (len(problems) == 0, problems)


def eval_dflash_on_box(host, port, pr_ref: str):
    """Run PR accuracy+bench then main bench with same prompt ids. Returns result dict."""
    print(f">> DFlash eval on box: PR ref={pr_ref}")
    r = ssh_run(host, port, _remote_script(pr_ref, do_accuracy=True, prompt_ids=None,
                                           n_tokens=BENCH_TOKENS))
    if r.returncode != 0:
        tail = ((r.stdout or "") + "\n" + (r.stderr or ""))[-2000:]
        return {"ok": False, "reason": "PR accuracy/bench failed", "log": tail}
    pr = _parse_remote(r.stdout or "")
    if "dflash_tps" not in pr:
        return {"ok": False, "reason": "PR bench missing DFLASH_TPS", "log": (r.stdout or "")[-1500:]}
    ids = pr.get("prompt_ids") or ""
    print(f">> PR DFlash={pr['dflash_tps']:.2f} AR={pr.get('ar_tps', 0):.2f} — measuring main …")
    r2 = ssh_run(host, port, _remote_script("main", do_accuracy=False, prompt_ids=ids,
                                            n_tokens=BENCH_TOKENS))
    if r2.returncode != 0:
        tail = ((r2.stdout or "") + "\n" + (r2.stderr or ""))[-2000:]
        return {"ok": False, "reason": "main bench failed", "log": tail, "pr": pr}
    main = _parse_remote(r2.stdout or "")
    if "dflash_tps" not in main:
        return {"ok": False, "reason": "main bench missing DFLASH_TPS", "log": (r2.stdout or "")[-1500:],
                "pr": pr}
    label, delta_pct, passed, reason = tier_from_gain(pr["dflash_tps"], main["dflash_tps"])
    guard_ok, guard_problems = check_qwen_guard(pr, main)
    if not guard_ok:
        # DFlash-only scoring is only valid alongside a clean Qwen3.5/3.6 guard — a regression
        # there overrides any DFlash tier, however good, into REJECT.
        label = "REJECT"
        passed = False
        reason = "qwen3.5/qwen3.6 no-regression guard failed: " + "; ".join(guard_problems[:6])
    return {
        "ok": True,
        "label": label,
        "pass": passed and label != "REJECT",
        "reason": reason,
        "delta_pct": delta_pct,
        "pr_dflash_tps": pr["dflash_tps"],
        "pr_ar_tps": pr.get("ar_tps"),
        "main_dflash_tps": main["dflash_tps"],
        "main_ar_tps": main.get("ar_tps"),
        "mean_accept": pr.get("mean_accept"),
        "spec_agree": pr.get("spec_agree"),
        "prompt_ids": ids,
        "speedup_vs_main": round(pr["dflash_tps"] / main["dflash_tps"], 3) if main["dflash_tps"] else 0,
        "speedup_vs_ar": round(pr["dflash_tps"] / pr["ar_tps"], 3) if pr.get("ar_tps") else 0,
        "guard_ok": guard_ok,
        "guard_problems": guard_problems,
    }


def format_comment(commit: str, res: dict) -> str:
    meta = {
        "label": res.get("label"),
        "delta_pct": res.get("delta_pct"),
        "pr_dflash_tps": res.get("pr_dflash_tps"),
        "main_dflash_tps": res.get("main_dflash_tps"),
        "pass": res.get("pass"),
    }
    marker = (
        f"<!-- sparkinfer-dflash-eval:{EVAL_SCHEMA_VERSION}:{commit} "
        f"{json.dumps(meta, separators=(',', ':'))} -->"
    )
    if not res.get("ok"):
        return (
            f"{marker}\n## sparkinfer dflash auto-eval — error\n\n"
            f"**reason:** `{res.get('reason')}`\n\n"
            f"<details><summary>log tail</summary>\n\n```\n{(res.get('log') or '')[:1800]}\n```\n</details>\n"
        )
    lab = res["label"]
    guard_ok = res.get("guard_ok")
    if guard_ok is None:
        guard_row = "| Qwen3.5/3.6 guard | — |\n"
    elif guard_ok:
        guard_row = "| Qwen3.5/3.6 guard | ✅ no regression (decode + prefill) |\n"
    else:
        problems = res.get("guard_problems") or []
        guard_row = "| Qwen3.5/3.6 guard | ❌ **REGRESSED** — DFlash score voided |\n"
        guard_row += "".join(f"| &nbsp;&nbsp;↳ | `{p}` |\n" for p in problems[:8])
    return (
        f"{marker}\n## sparkinfer dflash auto-eval — `eval-dflash:{lab}`\n\n"
        f"| metric | value |\n|---|---|\n"
        f"| **label** | `eval-dflash:{lab}` |\n"
        f"| PR DFlash tok/s | {res['pr_dflash_tps']:.2f} |\n"
        f"| main DFlash tok/s | {res['main_dflash_tps']:.2f} |\n"
        f"| speedup vs main | **{res['speedup_vs_main']:.2f}×** ({res['delta_pct']:+.1f}%) |\n"
        f"| PR AR tok/s | {res.get('pr_ar_tps') or 0:.2f} |\n"
        f"| DFlash vs AR | {res.get('speedup_vs_ar') or 0:.2f}× |\n"
        f"| mean accept τ | {res.get('mean_accept') or 0:.3f} |\n"
        f"| accuracy | {res.get('spec_agree') or 'VERDICT PASS'} |\n"
        f"{guard_row}"
        f"| commit | `{commit[:9]}` |\n\n"
        f"{res.get('reason') or ''}\n\n"
        "<sub>Scored on pinned RTX 5090 vs same-box `origin/main` DFlash, gated by a "
        "same-build Qwen3.5/3.6 decode+prefill no-regression guard. "
        "AR `eval:*` labels are frozen (casual bidir eval retired).</sub>\n"
    )


def auto_merge_ok_dflash(repo, num):
    info = json.loads(arb.gh([
        "pr", "view", str(num), "-R", repo, "--json",
        "state,isDraft,labels,author,mergeable,files",
    ]).stdout or "{}")
    if info.get("state") != "OPEN" or info.get("isDraft"):
        return False, "not an open, non-draft PR"
    labs = {l["name"] for l in info.get("labels", [])}
    tiers = {l.split(":", 1)[1] for l in labs if l.startswith(EVAL_PREFIX)}
    if not (tiers & SPEEDUP_LABELS):
        return False, "no verified eval-dflash:speedup label"
    if DFLASH_MERGE_FIRST not in labs:
        return False, "not dflash-merge-first"
    blocked = labs & AUTOMERGE_BLOCK
    if blocked:
        return False, f"blocking label(s): {', '.join(sorted(blocked))}"
    author = (info.get("author") or {}).get("login", "")
    if author.lower() in arb.load_denylist():
        return False, f"author {author} is blocked"
    if arb.author_penalty_until(author):
        return False, f"author {author} is under penalty"
    sens = [f["path"] for f in info.get("files", [])
            if any(f["path"].startswith(p) for p in arb.AUTOMERGE_SENSITIVE)]
    if sens:
        return False, f"touches protected paths: {', '.join(sens[:3])}"
    if arb.pr_merge_conflict(info.get("mergeable")):
        return False, "merge conflict with base"
    if info.get("mergeable") != "MERGEABLE":
        return False, f"not cleanly mergeable ({info.get('mergeable')})"
    return True, "ok"


def try_auto_merge_dflash(repo, num):
    ok, reason = auto_merge_ok_dflash(repo, num)
    if not ok:
        print(f">> dflash auto-merge SKIP #{num}: {reason}")
        return False
    r = arb.gh(["pr", "merge", str(num), "-R", repo, "--squash"])
    if r.returncode != 0 and os.environ.get("SPARKINFER_AUTOMERGE_ADMIN", "1") == "1":
        err = ((r.stderr or "") + (r.stdout or "")).lower()
        if "not mergeable" in err or "branch policy" in err or "required" in err or "prohibited" in err:
            print(">> dflash auto-merge: branch policy blocked — retrying with --admin")
            r = arb.gh(["pr", "merge", str(num), "-R", repo, "--squash", "--admin"])
    if r.returncode == 0:
        print(f">> DFLASH AUTO-MERGED #{num} (dflash-merge-first)")
        arb.gh(["pr", "comment", str(num), "-R", repo, "--body",
                "<!-- sparkinfer-dflash-automerge -->\n"
                "Auto-merged as the round's `dflash-merge-first` winner — "
                "verified same-box DFlash speedup over `main` with SPEC_AGREE pass."])
        return True
    print(f">> dflash auto-merge BLOCKED #{num}: {(r.stderr or r.stdout or '')[:200]}")
    return False


def reconcile_dflash_merge_labels(repo, dry_run=False):
    scores = _load_scores()
    open_prs = json.loads(arb.gh([
        "pr", "list", "-R", repo, "--state", "open",
        "--json", "number,labels", "--limit", "80",
    ]).stdout or "[]")
    open_labels = {p["number"]: {l["name"] for l in p["labels"]} for p in open_prs}

    # Clear dflash-merge-first from recently merged PRs
    merged = json.loads(arb.gh([
        "pr", "list", "-R", repo, "--state", "merged", "--label", DFLASH_MERGE_FIRST,
        "--json", "number", "--limit", "10",
    ]).stdout or "[]")
    for m in merged:
        if not dry_run:
            arb.remove_label(repo, m["number"], DFLASH_MERGE_FIRST)

    scored = []
    for num, labs in open_labels.items():
        if DFLASH_NEEDS_REBASE in labs:
            continue
        tiers = {l.split(":", 1)[1] for l in labs if l.startswith(EVAL_PREFIX)}
        tier = next((t for t in tiers if t in SPEEDUP_LABELS), None)
        if not tier:
            continue
        entry = scores.get(str(num)) or {}
        if entry.get("label") not in SPEEDUP_LABELS:
            # Prefer live label; use stored delta if present
            if tier not in SPEEDUP_LABELS:
                continue
            entry = {"label": tier, "delta_pct": entry.get("delta_pct") or 0}
        scored.append((num, float(entry.get("delta_pct") or 0), entry.get("label") or tier))

    scored.sort(key=lambda x: x[1], reverse=True)
    if not scored:
        print(">> dflash round: no verified speedup PRs")
        return
    winner = scored[0][0]
    print(f">> dflash round: merge-first #{winner}; rebase {[n for n,_,_ in scored[1:]] or 'none'}")
    if dry_run:
        return
    arb.add_label(repo, winner, DFLASH_MERGE_FIRST)
    arb.remove_label(repo, winner, DFLASH_NEEDS_REBASE)
    for num, _, _ in scored[1:]:
        arb.add_label(repo, num, DFLASH_NEEDS_REBASE)
        arb.remove_label(repo, num, DFLASH_MERGE_FIRST)
    if AUTO_MERGE:
        try_auto_merge_dflash(repo, winner)


def apply_result(repo, num, commit, res, dry_run=False):
    body = format_comment(commit, res)
    label = res.get("label") if res.get("ok") else "REJECT"
    if not res.get("ok"):
        label = "REJECT"
    print(f"PR #{num}: eval-dflash:{label}  "
          f"PR={res.get('pr_dflash_tps')} main={res.get('main_dflash_tps')} "
          f"delta={res.get('delta_pct')}%")
    if dry_run:
        print(body[:500])
        return
    strip_dflash_eval_labels(repo, num)
    arb.add_label(repo, num, f"{EVAL_PREFIX}{label}")
    arb.gh(["pr", "comment", str(num), "-R", repo, "--body", body])
    if res.get("ok") and res.get("delta_pct") is not None:
        scores = _load_scores()
        scores[str(num)] = {
            "commit": commit,
            "label": label,
            "delta_pct": res.get("delta_pct"),
            "pr_dflash_tps": res.get("pr_dflash_tps"),
            "main_dflash_tps": res.get("main_dflash_tps"),
            "pass": res.get("pass"),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _save_scores(scores)


def main():
    ap = argparse.ArgumentParser(description="DFlash PR eval bot")
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--repo", default="gittensor-ai-lab/sparkinfer")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reeval", action="store_true")
    ap.add_argument("--labels-only", action="store_true",
                    help="reconcile dflash-merge-first only — no GPU")
    ap.add_argument("--only-prs", default="",
                    help="comma-separated PR numbers (bypass greenlight; still require dflash paths unless forced)")
    ap.add_argument("--force-prs", action="store_true",
                    help="with --only-prs, evaluate even if paths do not match dflash filter")
    args = ap.parse_args()

    only = {int(x) for x in args.only_prs.split(",") if x.strip().isdigit()}

    print(f">> dflash eval transport: "
          f"{'ssh' if ssh_box_enabled() else f'vast.ai (instance {arb.current_instance(args.instance) or args.instance})'}")
    print(f">> AUTOMERGE={int(AUTO_MERGE)}")

    if args.labels_only:
        reconcile_dflash_merge_labels(args.repo, dry_run=args.dry_run)
        print("done — dflash labels only (no GPU).")
        return

    prs = json.loads(arb.gh([
        "pr", "list", "-R", args.repo, "--state", "open",
        "--json", "number,title,labels,isDraft,headRefOid,headRefName,mergeable,author,body",
        "--limit", "80",
    ]).stdout or "[]")
    prs.sort(key=lambda p: p["number"])

    pending = []
    for pr in prs:
        num = pr["number"]
        if only and num not in only:
            continue
        if pr.get("isDraft"):
            continue
        labs = {l["name"] for l in pr.get("labels", [])}
        if arb.HOLD_LABEL in labs:
            print(f"PR #{num}: hold — skip")
            continue
        if not only or not args.force_prs:
            if not is_dflash_pr(args.repo, num):
                if only:
                    print(f"PR #{num}: not a dflash-path PR — skip (use --force-prs)")
                continue
        else:
            if not is_dflash_pr(args.repo, num):
                print(f"PR #{num}: force — evaluating despite non-dflash paths")

        head = (pr.get("headRefOid") or "")[:40]
        short = head[:9]
        if not args.reeval and head and head in dflash_evaluated_commits(args.repo, num):
            print(f"PR #{num} @ {short}: already dflash-evaluated — skip")
            continue
        if arb.pr_merge_conflict(pr.get("mergeable")):
            print(f"PR #{num}: merge conflict — dflash-needs-rebase")
            if not args.dry_run:
                arb.add_label(args.repo, num, DFLASH_NEEDS_REBASE)
            continue

        if not only:
            status, why = arb.greenlight_status(args.repo, num, labs)
            if status != "ok":
                print(f"PR #{num}: not greenlit ({why}) — skip dflash eval")
                continue
            print(f"PR #{num}: greenlit ({why})")
        else:
            print(f"PR #{num}: --only-prs targeted")

        ref = f"pull/{num}/head"
        # Same-repo branches can use headRefName; pull/N/head always works.
        pending.append((num, head, short, ref))

    if not pending:
        reconcile_dflash_merge_labels(args.repo, dry_run=args.dry_run)
        print("done — no dflash PRs to evaluate.")
        return

    if args.dry_run:
        print("--- dry-run would evaluate: " + ", ".join(f"#{n}" for n, *_ in pending))
        return

    # Pin instance file
    pin = arb.PINNED_INSTANCE
    if pin and not ssh_box_enabled():
        with open(arb.INSTANCE_FILE, "w") as f:
            f.write(str(pin))

    try:
        host, port = resolve_ssh(args.instance)
    except Exception as e:
        print(f">> GPU unavailable: {e}")
        reconcile_dflash_merge_labels(args.repo, dry_run=False)
        print("done — dflash labels only (GPU down).")
        return

    print(f">> SSH root@{host}:{port}")

    for num, head, short, ref in pending:
        print(f"PR #{num} @ {short}: evaluating DFlash '{ref}' …")
        try:
            res = eval_dflash_on_box(host, port, ref)
        except Exception as e:
            res = {"ok": False, "reason": f"exception: {e}"}
        apply_result(args.repo, num, head or short, res, dry_run=False)

    reconcile_dflash_merge_labels(args.repo, dry_run=False)
    print("done — dflash eval pass complete.")


if __name__ == "__main__":
    main()
