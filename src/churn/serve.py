"""Executa FastAPI e Streamlit na mesma imagem e encerra ambos em caso de falha."""

import os
import signal
import subprocess
import sys
import time


def main():
    bind_host = os.environ.get("CHURN_BIND_HOST", "127.0.0.1")
    mode = os.environ.get("CHURN_SERVING_MODE", "online")
    if mode not in {"online", "batch"}:
        raise ValueError("CHURN_SERVING_MODE deve ser online ou batch.")
    module = "churn.app:app" if mode == "online" else "churn.api_lookup:app"
    processes = []

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    module,
                    "--host",
                bind_host,
                    "--port",
                    "8000",
                ]
            )
        )
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    "streamlit_app.py",
                    "--server.address",
                bind_host,
                    "--server.port",
                    "8501",
                ]
            )
        )
        while all(process.poll() is None for process in processes):
            time.sleep(0.5)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
