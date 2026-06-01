# import libs
import time
import requests
import os
import sys
from pathlib import Path
import json
import dotenv

# Variable definition
dotenv.load_dotenv()

model = sys.argv[1]
engine = sys.argv[2]
apiUrl = "http://localhost:5000/data"

outputDir = Path(os.getenv("PROJECT_DIR", ".")) / "logs" / "benchmark"
outputDir.mkdir(parents=True, exist_ok=True)

# if the model is a ".gguf" swap the end of its name with "GGUF" to prevent issues
if ".gguf" in model :
    model = model[:-5]
    model = model + "GGUF"
outputFile = outputDir / f"{model}_{engine}.json"

history = []

# retrieve data and timestamp from the API every second and when the tests are ended, saves them in a .json file
try:
    while True:
        try:
            res = requests.get(apiUrl, timeout=0.5).json()
            history.append({"timestamp": time.time(), "metrics": res})
            time.sleep(1.0)
        except:
            pass
except KeyboardInterrupt:
    with open(outputFile, "w") as f:
        json.dump(history, f, indent=2)