# lib import
import subprocess
import time

# GPU Parameters for the nvidia-smi command
gpuParams = [
    "name", 
    "pstate", 
    "utilization.gpu", 
    "utilization.memory",
    "memory.used", 
    "memory.total", 
    "temperature.gpu", 
    "temperature.memory",
    "power.draw", 
    "power.limit", 
    "clocks_event_reasons.hw_thermal_slowdown",
    "clocks.current.graphics", 
    "clocks.current.memory",
    "clocks.current.sm", 
    "pcie.link.gen.current", 
    "pcie.link.width.current",
    "ecc.errors.uncorrected.aggregate.total"
    ]


def getJobId():
    try:
        result = subprocess.check_output(
            ["squeue", "--me", "-h", "-S", "V", "-o", "%i"],
            stderr=subprocess.STDOUT
        ).decode('utf-8').strip().split('\n')
        return result[0]
    except Exception as e:
        print(f"Job ID recuperation failed : {e}")
        return None

def getModel(jobId):
    command = ["srun", "--jobid", str(jobId), "--overlap", "curl", "-s", "-H", "Content-Type: application/json", "http://localhost:8080/v1/models"]
    try:
        result = subprocess.check_output(command).decode('utf-8')
        data = json.loads(result)
        return data['data'][0]['id']
    except:
        return "Unknown Model"

def getServerTime(jobId):
    # PID Recuperation
    commandPid = ["srun", "--jobid", str(jobId), "--overlap", "pgrep", "-f", "-o", "server|vllm|python"]
    pid = subprocess.check_output(commandPid).decode().strip()
    pid = pid.split('\n')[0]

    # Launch timestamp recuperation
    commandLaunch = ["squeue", "-j", str(jobId), "-h", "-o", "%S"]
    launchTime = subprocess.check_output(commandLaunch).decode().strip().replace('T', ' ')

    # Server start timestamp recuperation
    commandReady = [
            "srun", 
            "--jobid", str(jobId), 
            "--overlap", 
            "stat", "-c", "%y", f"/proc/{pid}"
        ]
    readyTime = subprocess.check_output(commandReady).decode().strip().split('.')[0]

    return launchTime, readyTime

def getMetricsGpu(jobId):
    query = ",".join(gpuParams)
    command = [
        "srun", 
        "--jobid", str(jobId), 
        "--overlap",
        "nvidia-smi", 
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits"
    ]
    try :
        result = subprocess.check_output(command, stderr=subprocess.STDOUT, timeout=5).decode('utf-8').strip()
        gpu_lines = result.split('\n')
        
        all_gpus = []
        for line in gpu_lines:
            metrics = dict(zip(gpuParams, [r.strip() for r in line.split(',')]))
            all_gpus.append(metrics)
        return all_gpus
    except Exception as e:
        print(f"Error : {e}")
        return None


def getLLMMetrics(jobId):
    command = [
        "srun", "--jobid", str(jobId), "--overlap", 
        "curl", "-s", "http://localhost:8080/metrics"
    ]

    try :
        result = subprocess.check_output(command, stderr=subprocess.STDOUT, timeout=2).decode('utf-8')
        data = {}
        for line in result.strip().split('\n'):
            if not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    clean_key = parts[0].split('{')[0]
                    data[clean_key] = float(parts[1])
        return data

    except Exception as e:
        print(f"Error: {e}")
        return None