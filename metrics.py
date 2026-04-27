# lib import
import subprocess
import os
import time
import json

# constants
node = os.getenv('SLURMD_NODENAME', 'localhost')

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

jobId = getJobId()

# gpu params 
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
query = ",".join(gpuParams)


def getMetricsGpu():
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
        return dict(zip(gpuParams, [r.strip() for r in result.split(',')]))
    except Exception as e:
        print(f"Error : {e}")
        return None

def getMetricsLlama():
    command = [
        "srun", "--jobid", str(jobId), "--overlap", 
        "curl", "-s", "http://localhost:8080/metrics"
    ]

    try :
        result = subprocess.check_output(command, stderr=subprocess.STDOUT, timeout=2).decode('utf-8')

        lines = result.strip().split('\n')
        data = {}
        for line in lines:
            if not (line.startswith('#')):
                key, val = line.split()
                data[key] = float(val)
            
        return {
            "prompt_tokens_total":      data.get("llamacpp:prompt_tokens_total", 0),
            "prompt_seconds_total":     data.get("llamacpp:prompt_seconds_total", 0),
            "tokens_predicted_total":   data.get("llamacpp:tokens_predicted_total", 0),
            "tokens_predicted_seconds_total":  data.get("llamacpp:tokens_predicted_seconds_total", 0),
            
            "n_decode_total":       data.get("llamacpp:n_decode_total", 0),
            "n_tokens_max":   data.get("llamacpp:n_tokens_max", 0),
            "n_busy_slots_per_decode":     data.get("llamacpp:n_busy_slots_per_decode", 0),
            
            "prompt_tokens_seconds":     data.get("llamacpp:prompt_tokens_seconds", 0),
            "predicted_tokens_seconds":        data.get("llamacpp:predicted_tokens_seconds", 0),
            
            "req_processing": data.get("llamacpp:requests_processing", 0),
            "req_deferred":   data.get("llamacpp:requests_deferred", 0)
        }
    except Exception as e:
        print(f"Llama Error: {e}")
        return None

def getModel ():
    command = ["srun", "--jobid", str(jobId), "--overlap", "curl", "-s", "-H", "Content-Type: application/json", "http://localhost:8080/v1/models"]
    try:
        result = subprocess.check_output(command).decode('utf-8')
        data = json.loads(result)
        return data['data'][0]['id']
    except:
        return "Unknown Model"

def getServerTime():
    # PID Recuperation
    commandPid = ["srun", "--jobid", str(jobId), "--overlap", "pgrep", "llama-server"]
    pid = subprocess.check_output(commandPid).decode().strip()

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
            


def monitor():
    try:

        timestamps = getServerTime()
        model = getModel()

        while True: 
            gpuMetrics = getMetricsGpu()
            llamaMetrics = getMetricsLlama()
            os.system('clear')

            print(f"Metrics monitoring | Job : {jobId}")
            print(f"Started at {timestamps[0]} | Launched at {timestamps[1]}")

            if gpuMetrics is not None:
                # Display metrics
                print(f"\n GPU : {gpuMetrics['name']}")

                print(f"\n Temperature and energy consumption")
                print(f"  GPU Temperature      : {gpuMetrics['temperature.gpu']}°C")
                print(f"  Memory Temperature   : {gpuMetrics['temperature.memory']}°C")
                print(f"  GPU Thermal Slowdown : {gpuMetrics['clocks_event_reasons.hw_thermal_slowdown']}")
                print(f"  Consumption          : {gpuMetrics['power.draw']}W / {gpuMetrics['power.limit']}W")

                print(f"\n Usage")
                print(f"  GPU Kernel Usage     : {gpuMetrics['utilization.gpu']}%")
                print(f"  VRAM Usage           : {gpuMetrics['memory.used']} MB / {gpuMetrics['memory.total']} MB")

                print(f"\n Clocks")
                print(f"  Graphical clock      : {gpuMetrics['clocks.current.graphics']} MHz / {gpuMetrics['clocks.current.sm']} MHz")
                print(f"  Memory clock         : {gpuMetrics['clocks.current.memory']} MHz")
                
            
            else : 
                return print(f"FAILED : GPU Metrics")

            if llamaMetrics:
                print(f"\n Llama : {model}")

                print(f"\n Request")
                print(f"  Running : {llamaMetrics['req_processing']}, Pending : {llamaMetrics['req_deferred']}")

                print(f"\n Request Stats ")
                print(f"  Prompt process time  : {llamaMetrics['prompt_seconds_total']:.2f} t/s")
                print(f"  Tokens processed     : {llamaMetrics['prompt_tokens_total']} tokens")
                print(f"  Predict process time : {llamaMetrics['tokens_predicted_seconds_total']:.3f} s")
                print(f"  Tokens Generated     : {llamaMetrics['tokens_predicted_total']} tokens")
                print(f"  Tokens In/Out        : {llamaMetrics['prompt_tokens_total']} / {llamaMetrics['tokens_predicted_total']}")
                
                print(f"\n Tokens Stats")
                print(f"  Number of Decode     : {llamaMetrics['n_decode_total']}")
                print(f"  Max Tokens Observed  : {llamaMetrics['n_tokens_max']}")
                print(f"  Busy Slots Average   : {llamaMetrics['n_busy_slots_per_decode']:.2f}")

            else:
                print(f"FAILED : Llama Metrics")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n Fin du monitoring.")

        
monitor()