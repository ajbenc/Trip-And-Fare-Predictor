"""Unified launcher for FastAPI API and Streamlit UI.
Usage (Windows PowerShell):
  python start_all.py

This script starts both services concurrently and waits until either exits.
It attempts a graceful shutdown on Ctrl+C.
"""
from __future__ import annotations
import subprocess, sys, time, signal, os
from threading import Thread

FASTAPI_CMD = [sys.executable, "-m", "uvicorn", "src.interface.api.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
STREAMLIT_CMD = [sys.executable, "-m", "streamlit", "run", "src/interface/web/streamlit_app.py", "--server.port", "8501"]

processes: list[subprocess.Popen] = []

stop_requested = False

def _stream_output(proc: subprocess.Popen, name: str):
    for line in proc.stdout:  # type: ignore
        sys.stdout.write(f"[{name}] {line.decode(errors='replace')}")
    sys.stdout.flush()

def launch(name: str, cmd: list[str]):
    env = os.environ.copy()
    # Prefer utf-8
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    processes.append(proc)
    t = Thread(target=_stream_output, args=(proc, name), daemon=True)
    t.start()
    return proc

def graceful_shutdown():
    global stop_requested
    if stop_requested:
        return
    stop_requested = True
    print("\n[launcher] Shutting down services...")
    for p in processes:
        if p.poll() is None:
            try:
                p.send_signal(signal.SIGINT)
            except Exception:
                pass
    # Wait a little then force kill any stragglers
    time.sleep(3)
    for p in processes:
        if p.poll() is None:
            try:
                p.kill()
            except Exception:
                pass
    print("[launcher] All services stopped.")

def main():
    print("[launcher] Starting FastAPI (8000) and Streamlit (8501)...")
    fastapi = launch("fastapi", FASTAPI_CMD)
    streamlit = launch("streamlit", STREAMLIT_CMD)

    print("[launcher] Waiting for health endpoint...")
    for _ in range(15):
        if fastapi.poll() is not None:
            print("[launcher] FastAPI process exited early.")
            break
        try:
            import urllib.request
            with urllib.request.urlopen("http://localhost:8000/health", timeout=2) as r:
                if r.status == 200:
                    print("[launcher] FastAPI health OK.")
                    break
        except Exception:
            pass
        time.sleep(1)
    print("[launcher] UI: http://localhost:8501  |  API: http://localhost:8000/docs")

    def _sigint_handler(signum, frame):
        graceful_shutdown()
        sys.exit(0)
    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)

    # Monitor processes
    while True:
        all_exited = all(p.poll() is not None for p in processes)
        if all_exited:
            print("[launcher] One or more services terminated; exiting.")
            break
        time.sleep(2)

    graceful_shutdown()

if __name__ == "__main__":
    main()
