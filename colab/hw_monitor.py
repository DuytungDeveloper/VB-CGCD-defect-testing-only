#!/usr/bin/env python
"""Lightweight hardware monitor: tracks CPU %, RAM GB, GPU % and GPU VRAM MB peaks."""
import argparse, json, signal, subprocess, time
from pathlib import Path

STOP = False


def _stop(_sig, _frm):
    global STOP
    STOP = True


def _read_cpu():
    vals = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
    return sum(vals), idle


def _read_ram_gb():
    info = {}
    for ln in Path("/proc/meminfo").read_text().splitlines():
        k, v = ln.split(":", 1)
        info[k] = int(v.strip().split()[0])
    used_kb = info["MemTotal"] - info.get("MemAvailable", info.get("MemFree", 0))
    return used_kb / 1024.0 / 1024.0


def _read_gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL
        ).strip().splitlines()[0]
        u, m = [float(x.strip()) for x in out.split(",")]
        return u, m
    except Exception:
        return 0.0, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=2.0)
    o = ap.parse_args()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    prev_t, prev_i = _read_cpu()
    cpu_samples = []
    ram_peak, gpu_u_peak, gpu_m_peak = _read_ram_gb(), 0.0, 0.0
    gu, gm = _read_gpu()
    gpu_u_peak, gpu_m_peak = max(gpu_u_peak, gu), max(gpu_m_peak, gm)

    while not STOP:
        time.sleep(o.interval)
        cur_t, cur_i = _read_cpu()
        dt, di = cur_t - prev_t, cur_i - prev_i
        if dt > 0:
            cpu_samples.append(100.0 * (dt - di) / dt)
        prev_t, prev_i = cur_t, cur_i
        ram_peak = max(ram_peak, _read_ram_gb())
        gu, gm = _read_gpu()
        gpu_u_peak, gpu_m_peak = max(gpu_u_peak, gu), max(gpu_m_peak, gm)

    res = {
        "cpu_peak_pct": round(max(cpu_samples) if cpu_samples else 0.0, 1),
        "cpu_avg_pct": round(sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0, 1),
        "ram_peak_gb": round(ram_peak, 2),
        "gpu_util_peak_pct": round(gpu_u_peak, 1),
        "gpu_vram_peak_mb": round(gpu_m_peak, 1),
    }
    Path(o.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
