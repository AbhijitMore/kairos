from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
import uvicorn
import time
import threading
import requests

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


Instrumentator().instrument(app).expose(app)


def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8081)


if __name__ == "__main__":
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    time.sleep(2)
    try:
        r = requests.get("http://127.0.0.1:8081/metrics")
        print(f"Status: {r.status_code}")
        print(r.text[:100])
    except Exception as e:
        print(f"Error: {e}")
