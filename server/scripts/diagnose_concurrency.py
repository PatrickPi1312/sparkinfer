#!/usr/bin/env python3
"""Reproduce and capture short-stream concurrency scaling through the HTTP API.

Stdlib-only. The JSON report contains the exact requests, externally measured timing,
server usage, capacity samples, GPU telemetry, PCIe state, and optional Docker limits.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_CONCURRENCY = (6, 8, 10, 12)
GPU_FIELDS = (
    "timestamp,index,name,utilization.gpu,utilization.memory,"
    "clocks.current.graphics,clocks.current.memory,power.draw,temperature.gpu,"
    "memory.used,memory.total,pcie.link.gen.current,pcie.link.width.current"
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def request_json(url: str, body: dict[str, Any] | None = None, timeout: float = 10) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"} if data else {}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def command(args: list[str], timeout: float = 30) -> dict[str, Any]:
    try:
        run = subprocess.run(args, text=True, capture_output=True, timeout=timeout, check=False)
        return {"argv": args, "returncode": run.returncode, "stdout": run.stdout, "stderr": run.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": args, "error": str(exc)}


def docker_details(container: str) -> dict[str, Any]:
    if not container or not shutil.which("docker"):
        return {}
    raw = command(["docker", "inspect", container])
    try:
        item = json.loads(raw.get("stdout", "[]"))[0]
    except (IndexError, KeyError, json.JSONDecodeError):
        return {"inspect_error": raw}
    env = {}
    for entry in item.get("Config", {}).get("Env", []):
        key, _, value = entry.partition("=")
        if key.startswith("SPARKINFER_") or key in {
            "CTX", "MODEL_FILE", "MODEL_NAME", "MODEL_SHA256", "NVIDIA_VISIBLE_DEVICES"
        }:
            env[key] = value
    host = item.get("HostConfig", {})
    return {
        "id": item.get("Id"),
        "image": item.get("Config", {}).get("Image"),
        "image_id": item.get("Image"),
        "environment": env,
        "limits": {
            "NanoCpus": host.get("NanoCpus"),
            "CpuQuota": host.get("CpuQuota"),
            "CpuPeriod": host.get("CpuPeriod"),
            "CpusetCpus": host.get("CpusetCpus"),
            "Memory": host.get("Memory"),
            "MemorySwap": host.get("MemorySwap"),
            "PidsLimit": host.get("PidsLimit"),
        },
    }


def make_payload(model: str, request_id: int, max_tokens: int, logprobs: bool) -> dict[str, Any]:
    # Distinct, stable prompts prevent a proxy from coalescing requests while keeping prefill short.
    prompt = (
        f"Concurrency diagnostic request {request_id:02d}. "
        "Write a compact technical explanation of why deterministic inference matters. "
        "Continue until the token limit."
    )
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "logprobs": logprobs,
        "enable_thinking": False,
    }
    if logprobs:
        payload["top_logprobs"] = 1
    return payload


def run_stream(base_url: str, payload: dict[str, Any], request_id: int, timeout: float) -> dict[str, Any]:
    started_utc = utc_now()
    started = time.perf_counter()
    first_event_s = None
    content = []
    reasoning = []
    logprob_entries = 0
    finish_reason = None
    usage = None
    status = None
    error = None
    try:
        req = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status = response.status
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if "error" in event:
                    error = str(event["error"])
                    break
                choices = event.get("choices", [])
                if choices:
                    choice = choices[0]
                    delta = choice.get("delta", {})
                    if delta.get("content") is not None:
                        content.append(delta.get("content", ""))
                    if delta.get("reasoning_content") is not None:
                        reasoning.append(delta.get("reasoning_content", ""))
                    entries = (choice.get("logprobs") or {}).get("content") or []
                    logprob_entries += len(entries)
                    if (delta.get("content") or delta.get("reasoning_content") or entries) and first_event_s is None:
                        first_event_s = time.perf_counter() - started
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
                if event.get("usage") is not None:
                    usage = event["usage"]
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = exc.read().decode("utf-8", errors="replace")[:1000]
    except Exception as exc:  # diagnostic report must retain every failed request
        error = f"{type(exc).__name__}: {exc}"
    wall_s = time.perf_counter() - started
    return {
        "request_id": request_id,
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "http_status": status,
        "wall_s": wall_s,
        "external_ttft_s": first_event_s,
        "finish_reason": finish_reason,
        "usage": usage,
        "logprob_entries": logprob_entries,
        "content": "".join(content),
        "reasoning_content": "".join(reasoning),
        "error": error,
    }


class Sampler:
    def __init__(self, base_url: str, interval: float):
        self.base_url = base_url
        self.interval = interval
        self.stop = threading.Event()
        self.capacity: list[dict[str, Any]] = []
        self.gpu: list[dict[str, Any]] = []
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self.stop.is_set():
            stamp = utc_now()
            try:
                value = request_json(f"{self.base_url}/v1/capacity", timeout=2)
                self.capacity.append({"timestamp": stamp, "value": value})
            except Exception as exc:
                self.capacity.append({"timestamp": stamp, "error": str(exc)})
            if shutil.which("nvidia-smi"):
                sample = command(
                    ["nvidia-smi", f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader,nounits"],
                    timeout=3,
                )
                self.gpu.append({"timestamp": stamp, **sample})
            self.stop.wait(self.interval)

    def __enter__(self) -> "Sampler":
        self.thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop.set()
        self.thread.join(timeout=max(2.0, self.interval * 2))


def run_burst(
    base_url: str,
    model: str,
    concurrency: int,
    max_tokens: int,
    logprobs: bool,
    timeout: float,
    sample_interval: float,
) -> dict[str, Any]:
    payloads = [make_payload(model, i, max_tokens, logprobs) for i in range(concurrency)]
    tokenized = []
    for payload in payloads:
        try:
            tokenized.append(request_json(f"{base_url}/v1/tokenize", payload, timeout=10))
        except Exception as exc:
            tokenized.append({"error": str(exc)})
    before = request_json(f"{base_url}/v1/capacity")
    started_utc = utc_now()
    started = time.perf_counter()
    with Sampler(base_url, sample_interval) as samples:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(run_stream, base_url, payload, i, timeout)
                for i, payload in enumerate(payloads)
            ]
            results = [future.result() for future in futures]
    wall_s = time.perf_counter() - started
    after = request_json(f"{base_url}/v1/capacity")
    completion_tokens = sum(int((item.get("usage") or {}).get("completion_tokens", 0)) for item in results)
    successful = sum(item.get("http_status") == 200 and not item.get("error") for item in results)
    return {
        "concurrency": concurrency,
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "wall_s": wall_s,
        "successful_requests": successful,
        "completion_tokens": completion_tokens,
        "external_aggregate_completion_tps": completion_tokens / wall_s if wall_s else None,
        "capacity_before": before,
        "capacity_during": samples.capacity,
        "capacity_after": after,
        "gpu_during": samples.gpu,
        "payloads": payloads,
        "tokenize_responses": tokenized,
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="qwen3.6-35b-a3b")
    parser.add_argument("--concurrency", default="6,8,10,12", help="comma-separated levels")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--no-logprobs", action="store_true")
    parser.add_argument("--container", default="sparkinfer", help="Docker container name; empty disables")
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    levels = [int(value) for value in args.concurrency.split(",") if value.strip()]
    output = args.output or Path(f"sparkinfer-concurrency-{dt.datetime.now():%Y%m%d-%H%M%S}.json")
    request_json(f"{base_url}/health", timeout=5)
    report: dict[str, Any] = {
        "schema": 1,
        "created_utc": utc_now(),
        "command": {
            "base_url": base_url,
            "model": args.model,
            "concurrency": levels,
            "max_tokens": args.max_tokens,
            "repeats": args.repeats,
            "logprobs": not args.no_logprobs,
            "sample_interval": args.sample_interval,
        },
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "sparkinfer_environment": {
                key: value for key, value in os.environ.items() if key.startswith("SPARKINFER_")
            },
        },
        "server": {
            "models": request_json(f"{base_url}/v1/models"),
            "info": request_json(f"{base_url}/v1/info"),
            "capacity": request_json(f"{base_url}/v1/capacity"),
        },
        "container": docker_details(args.container),
        "nvidia_smi_query": command(["nvidia-smi"], timeout=15) if shutil.which("nvidia-smi") else {},
        "nvidia_smi_q_before": command(["nvidia-smi", "-q"], timeout=30) if shutil.which("nvidia-smi") else {},
        "bursts": [],
    }
    for repeat in range(args.repeats):
        for concurrency in levels:
            print(f"repeat={repeat + 1}/{args.repeats} concurrency={concurrency}", flush=True)
            burst = run_burst(
                base_url, args.model, concurrency, args.max_tokens, not args.no_logprobs,
                args.timeout, args.sample_interval,
            )
            burst["repeat"] = repeat + 1
            report["bursts"].append(burst)
            print(
                f"  {burst['successful_requests']}/{concurrency} ok, "
                f"{burst['completion_tokens']} tokens / {burst['wall_s']:.3f}s = "
                f"{burst['external_aggregate_completion_tps']:.2f} tok/s",
                flush=True,
            )
    report["nvidia_smi_q_after"] = (
        command(["nvidia-smi", "-q"], timeout=30) if shutil.which("nvidia-smi") else {}
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
