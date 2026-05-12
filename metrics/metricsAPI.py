from fastapi import FastAPI
import uvicorn
import os
import time
from getMetrics import getLLMMetrics, getMetricsGpu, getJobId

app = FastAPI(title="LLM Metrics")

@app.get("/data")
async def all_data():
    return {
        "job_id": getJobId(),
        "node": os.getenv('HOSTNAME', 'localhost'),
        "timestamp": time.time(),
        "gpu": getMetricsGpu(getJobId()),
        "llm": getLLMMetrics(getJobId())
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)