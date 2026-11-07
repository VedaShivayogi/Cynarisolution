"""
Load test for the FinWise API.

    python loadtest/loadtest.py --client httpx        # realistic app-style client (heavy)   -> docs/loadtest_results_httpx.json
    python loadtest/loadtest.py --client lean         # raw-socket client, isolates the server  -> docs/loadtest_results_lean.json
    python loadtest/loadtest.py --quick               # ~30 s smoke version
    python loadtest/loadtest.py --url http://host:8000 --only-sustained   # point at an already-running server

Scenarios (all use real rows from finwise_loan_applications.csv, random per request, fixed seed)
  burst     50 requests fired at the same instant, repeated N rounds. Answers: "what does a user see when 50
            requests land together?"  (latency measured per request from the moment the burst starts)
  sustained 50 virtual users, each sending the next request as soon as the last one returns, for T seconds.
            Answers: "what throughput can we hold with 50 concurrent requests in flight, and at what latency?"
  ramp      1, 10, 25, 50, 100 virtual users, a few seconds each. Shows where the service saturates.

NOTE: the load generator runs on the same machine as the server. Results include that contention, so they are
conservative; on separate hosts throughput will be higher.
"""
import argparse, asyncio, json, os, platform, random, signal, statistics as st, subprocess, sys, time
from pathlib import Path

import httpx, pandas as pd
import psutil

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ["age", "income_annual_inr", "employment_type", "employment_years", "existing_loans", "existing_emi_inr",
            "credit_score", "loan_amount_inr", "loan_purpose", "ltv_ratio", "dti_ratio"]


def load_payloads():
    df = pd.read_csv(ROOT / "finwise_loan_applications.csv")
    out = []
    for _, r in df.iterrows():
        d = {k: r[k] for k in FEATURES}
        for k in ("age", "existing_loans", "credit_score"): d[k] = int(d[k])
        for k in ("income_annual_inr", "employment_years", "existing_emi_inr", "loan_amount_inr", "ltv_ratio", "dti_ratio"): d[k] = float(d[k])
        out.append(d)
    return out


def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def summarize(lat_ms, srv_ms, errors, wall_s, n_ok):
    return {"requests": n_ok + errors, "ok": n_ok, "errors": errors, "wall_s": round(wall_s, 2),
            "throughput_rps": round(n_ok / wall_s, 1) if wall_s else 0,
            "client_ms": {"mean": round(st.mean(lat_ms), 1) if lat_ms else None, "p50": round(pct(lat_ms, .5), 1),
                          "p95": round(pct(lat_ms, .95), 1), "p99": round(pct(lat_ms, .99), 1), "max": round(max(lat_ms), 1) if lat_ms else None},
            "server_ms": {"p50": round(pct(srv_ms, .5), 1), "p95": round(pct(srv_ms, .95), 1), "p99": round(pct(srv_ms, .99), 1)}}


async def one(client, payloads, rng, acc):
    t0 = time.perf_counter()
    try:
        r = await client.post("/predict", json=rng.choice(payloads))
        dt = (time.perf_counter() - t0) * 1e3
        if r.status_code == 200:
            acc["lat"].append(dt); acc["srv"].append(float(r.headers.get("x-process-time-ms", "nan"))); acc["ok"] += 1
        else:
            acc["err"] += 1; acc["codes"][r.status_code] = acc["codes"].get(r.status_code, 0) + 1
    except Exception as e:                                        # timeouts, resets
        acc["err"] += 1; acc["codes"][type(e).__name__] = acc["codes"].get(type(e).__name__, 0) + 1


def new_acc(): return {"lat": [], "srv": [], "ok": 0, "err": 0, "codes": {}}


async def burst(url, payloads, size=50, rounds=20, seed=1):
    rng, acc = random.Random(seed), new_acc()
    lim = httpx.Limits(max_connections=size, max_keepalive_connections=size)
    async with httpx.AsyncClient(base_url=url, timeout=30, limits=lim) as c:
        await asyncio.gather(*[one(c, payloads, rng, new_acc()) for _ in range(size)])      # warm connections
        t0 = time.perf_counter()
        for _ in range(rounds):
            await asyncio.gather(*[one(c, payloads, rng, acc) for _ in range(size)])
        wall = time.perf_counter() - t0
    return {**summarize(acc["lat"], acc["srv"], acc["err"], wall, acc["ok"]), "size": size, "rounds": rounds, "error_codes": acc["codes"]}


async def sustained(url, payloads, users, seconds, seed=2):
    rng, acc = random.Random(seed), new_acc()
    lim = httpx.Limits(max_connections=users, max_keepalive_connections=users)
    async with httpx.AsyncClient(base_url=url, timeout=30, limits=lim) as c:
        await asyncio.gather(*[one(c, payloads, rng, new_acc()) for _ in range(min(users, 20))])   # warm-up
        stop = time.perf_counter() + seconds
        async def user():
            while time.perf_counter() < stop:
                await one(c, payloads, rng, acc)
        t0 = time.perf_counter()
        await asyncio.gather(*[user() for _ in range(users)])
        wall = time.perf_counter() - t0
    return {**summarize(acc["lat"], acc["srv"], acc["err"], wall, acc["ok"]), "users": users, "error_codes": acc["codes"]}



# ---- lean client: raw asyncio sockets + pre-serialised requests (~0.3 ms CPU/request instead of ~4.5 ms for httpx).
# Purpose: on a small box a heavy Python client starves the server of CPU, so httpx numbers understate the server.
def build_raw_requests(payloads):
    out = []
    for p in payloads:
        b = json.dumps(p).encode()
        out.append(b"POST /predict HTTP/1.1\r\nHost: bench\r\nContent-Type: application/json\r\nContent-Length: " + str(len(b)).encode() + b"\r\n\r\n" + b)
    return out


async def lean_one(reader, writer, req, acc):
    t0 = time.perf_counter()
    try:
        writer.write(req)
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 30)
        status = int(head[9:12])
        hl = head.lower()
        n = int(hl.split(b"content-length:")[1].split(b"\r\n")[0])
        await reader.readexactly(n)
        dt = (time.perf_counter() - t0) * 1e3
        if status == 200:
            acc["lat"].append(dt); acc["ok"] += 1
            i = hl.find(b"x-process-time-ms:")
            acc["srv"].append(float(hl[i + 18:hl.find(b"\r\n", i)]) if i >= 0 else float("nan"))
        else:
            acc["err"] += 1; acc["codes"][status] = acc["codes"].get(status, 0) + 1
    except Exception as e:
        acc["err"] += 1; acc["codes"][type(e).__name__] = acc["codes"].get(type(e).__name__, 0) + 1


async def _connect(url, n):
    host, port = url.split("//")[1].split(":")
    return [await asyncio.open_connection(host, int(port)) for _ in range(n)]


async def lean_burst(url, payloads, size=50, rounds=20, seed=1):
    rng, acc, reqs = random.Random(seed), new_acc(), build_raw_requests(payloads)
    conns = await _connect(url, size)
    await asyncio.gather(*[lean_one(r, w, rng.choice(reqs), new_acc()) for r, w in conns])
    t0 = time.perf_counter()
    for _ in range(rounds):
        await asyncio.gather(*[lean_one(r, w, rng.choice(reqs), acc) for r, w in conns])
    wall = time.perf_counter() - t0
    for _, w in conns: w.close()
    return {**summarize(acc["lat"], acc["srv"], acc["err"], wall, acc["ok"]), "size": size, "rounds": rounds, "error_codes": acc["codes"]}


async def lean_sustained(url, payloads, users, seconds, seed=2):
    rng, acc, reqs = random.Random(seed), new_acc(), build_raw_requests(payloads)
    conns = await _connect(url, users)
    await asyncio.gather(*[lean_one(r, w, rng.choice(reqs), new_acc()) for r, w in conns])
    stop = time.perf_counter() + seconds
    async def user(r, w):
        while time.perf_counter() < stop:
            await lean_one(r, w, rng.choice(reqs), acc)
    t0 = time.perf_counter()
    await asyncio.gather(*[user(r, w) for r, w in conns])
    wall = time.perf_counter() - t0
    for _, w in conns: w.close()
    return {**summarize(acc["lat"], acc["srv"], acc["err"], wall, acc["ok"]), "users": users, "error_codes": acc["codes"]}


def tree_cpu_s(pid):
    """CPU seconds (user+system) consumed so far by a process and all its children (uvicorn workers)."""
    p = psutil.Process(pid); total = 0.0
    for q in [p] + p.children(recursive=True):
        try: t = q.cpu_times(); total += t.user + t.system
        except psutil.Error: pass
    return total


IS_WIN = os.name == "nt"


def start_server(path, workers, port):
    """Start uvicorn as a child process. Server output goes to docs/server.log so a failed start is diagnosable."""
    env = {**os.environ, "FINWISE_SCORING_PATH": path}
    logf = ROOT / "docs" / "server.log"
    logf.parent.mkdir(exist_ok=True)
    log = open(logf, "w")
    kw = {} if IS_WIN else {"start_new_session": True}          # POSIX only: lets us kill the whole worker group
    p = subprocess.Popen([sys.executable, "-W", "ignore", "-m", "uvicorn", "serving.app:app", "--host", "127.0.0.1", "--port", str(port),
                          "--workers", str(workers), "--log-level", "warning"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, **kw)
    for _ in range(180):                                         # up to 90 s (Windows spawns workers slowly)
        if p.poll() is not None:
            break                                                # server process already died: stop waiting
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200: return p
        except httpx.HTTPError: time.sleep(0.5)
    stop_server(p); log.close()
    tail = "\n".join(logf.read_text(errors="replace").splitlines()[-15:])
    raise RuntimeError(f"server failed to start. Last lines of {logf}:\n{tail}\n"
                       f"Common causes: dependencies not installed (pip install -r requirements-dev.txt), port {port} already in use, "
                       f"or running from the wrong folder (run from the folder that contains serving/ and data_pipeline.py).")


def stop_server(p):
    if IS_WIN:                                                   # kill the whole process tree (uvicorn parent + workers)
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
    else:
        try: os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except ProcessLookupError: return
    try: p.wait(10)
    except subprocess.TimeoutExpired:
        if not IS_WIN: os.killpg(os.getpgid(p.pid), signal.SIGKILL)


def env_info():
    cpu = next((l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), platform.processor()) if Path("/proc/cpuinfo").exists() else platform.processor()
    import xgboost, sklearn, fastapi, uvicorn
    return {"cpu_model": cpu, "logical_cpus": os.cpu_count(), "python": platform.python_version(), "os": platform.platform(),
            "xgboost": xgboost.__version__, "scikit_learn": sklearn.__version__, "pandas": pd.__version__, "fastapi": fastapi.__version__, "uvicorn": uvicorn.__version__}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url"); ap.add_argument("--quick", action="store_true"); ap.add_argument("--only-sustained", action="store_true")
    ap.add_argument("--client", choices=["httpx", "lean"], default="httpx"); ap.add_argument("--out"); ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    a.out = a.out or str(ROOT / f"docs/loadtest_results_{a.client}.json")
    B, S = (lean_burst, lean_sustained) if a.client == "lean" else (burst, sustained)
    payloads = load_payloads()
    rounds, secs, ramp_secs = (4, 5, 3) if a.quick else (20, 20, 8)

    if a.url:
        print(json.dumps(asyncio.run(S(a.url, payloads, 50, secs)), indent=1)); return

    configs = [("pipeline", 1), ("fast", 1), ("fast", 2)]
    results = {"client": a.client, "env": env_info(), "settings": {"burst_rounds": rounds, "sustained_seconds": secs, "ramp_seconds": ramp_secs}, "runs": {}}
    for path, workers in configs:
        key = f"{path}_{workers}w"
        print(f"\n=== {key}", flush=True)
        srv = start_server(path, workers, a.port)
        try:
            url, run = f"http://127.0.0.1:{a.port}", {}
            run["burst_50"] = asyncio.run(B(url, payloads, 50, rounds)); print(" burst_50     ", run["burst_50"]["throughput_rps"], "rps  p95", run["burst_50"]["client_ms"]["p95"], "ms  err", run["burst_50"]["errors"], flush=True)
            c0, s0 = time.process_time(), tree_cpu_s(srv.pid)
            run["sustained_50"] = asyncio.run(S(url, payloads, 50, secs))
            cc, sc, w = time.process_time() - c0, tree_cpu_s(srv.pid) - s0, run["sustained_50"]["wall_s"]
            run["sustained_50"]["cpu"] = {"client_pct_of_one_core": round(100 * cc / w), "server_pct_of_one_core": round(100 * sc / w),
                                          "server_cpu_ms_per_request": round(1e3 * sc / run["sustained_50"]["requests"], 2),
                                          "client_cpu_ms_per_request": round(1e3 * cc / run["sustained_50"]["requests"], 2)}
            print(" sustained_50 ", run["sustained_50"]["throughput_rps"], "rps  p95", run["sustained_50"]["client_ms"]["p95"], "ms  err", run["sustained_50"]["errors"], run["sustained_50"]["cpu"], flush=True)
            if (path, workers) in (("pipeline", 1), ("fast", 1)):
                run["ramp"] = {}
                for u in (1, 10, 25, 50, 100):
                    run["ramp"][str(u)] = asyncio.run(S(url, payloads, u, ramp_secs)); print(f" ramp users={u:<3}", run["ramp"][str(u)]["throughput_rps"], "rps  p95", run["ramp"][str(u)]["client_ms"]["p95"], "ms", flush=True)
            results["runs"][key] = run
        finally:
            stop_server(srv); time.sleep(1)
    Path(a.out).parent.mkdir(exist_ok=True, parents=True)
    json.dump(results, open(a.out, "w"), indent=2)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
