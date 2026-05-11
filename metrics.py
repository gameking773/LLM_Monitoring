# lib import
import subprocess
import os
import time
import json

# constants
node = os.getenv('SLURMD_NODENAME', 'localhost')
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

jobId = getJobId()



def getMetricsGpu():
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

def getLLMMetrics():
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
            

def displayGpu():
    gpuMetrics = getMetricsGpu()
    if gpuMetrics is not None:
        # Display metrics for each gpu
        for i, gpu in enumerate(gpuMetrics):
            print(f"\n GPU {i} : {gpu['name']}")

            print(f"\n Temperature and energy consumption (GPU {i})")
            print(f"  GPU Temperature      : {gpu['temperature.gpu']}°C")
            print(f"  Memory Temperature   : {gpu['temperature.memory']}°C")
            print(f"  GPU Thermal Slowdown : {gpu['clocks_event_reasons.hw_thermal_slowdown']}")
            print(f"  Consumption          : {gpu['power.draw']}W / {gpu['power.limit']}W")

            print(f"\n Usage (GPU {i})")
            print(f"  GPU Kernel Usage     : {gpu['utilization.gpu']}%")
            print(f"  Memory Usage         : {gpu['utilization.memory']}%")
            print(f"  VRAM Usage           : {gpu['memory.used']} MB / {gpu['memory.total']} MB")

            print(f"\n Clocks (GPU {i})")
            print(f"  Graphical clock      : {gpu['clocks.current.graphics']} MHz / {gpu['clocks.current.sm']} MHz")
            print(f"  Memory clock         : {gpu['clocks.current.memory']} MHz")
            print("-" * 30)

    else : 
        return print(f"FAILED : GPU Metrics")


def displayLlama(data):
    model = getModel()
    
    if data:
        print(f"\n Llama : {model}")

        print(f"\n Request")
        print(f"  Running : {data['llamacpp:requests_processing']}, Pending : {data['llamacpp:requests_deferred']}")

        print(f"\n Request Stats ")
        print(f"  Prompt process time  : {data['llamacpp:prompt_seconds_total']:.2f} t/s")
        print(f"  Tokens processed     : {data['llamacpp:prompt_tokens_total']} tokens")
        print(f"  Predict process time : {data['llamacpp:tokens_predicted_seconds_total']:.3f} s")
        print(f"  Tokens Generated     : {data['llamacpp:tokens_predicted_total']} tokens")
        print(f"  Tokens In/Out        : {data['llamacpp:prompt_tokens_total']} / {data['llamacpp:tokens_predicted_total']}")
            
        print(f"\n Tokens Stats")
        print(f"  Number of Decode     : {data['llamacpp:n_decode_total']}")
        print(f"  Max Tokens Observed  : {data['llamacpp:n_tokens_max']}")
        print(f"  Busy Slots Average   : {data['llamacpp:n_busy_slots_per_decode']:.2f}")

    else:
        print(f"FAILED : Llama Metrics")

def displayVLLM(data):
    if data:
        print(f"\n Request")
        print(f"  Running : {data.get('vllm:num_requests_running')}, Pending : {data.get('vllm:num_requests_waiting')}")

        print(f"\n Request Stats ")
        print(f"  Tokens processed     : {data.get('vllm:prompt_tokens_total')} tokens")
        print(f"  Tokens Generated     : {data.get('vllm:generation_tokens_total')} tokens")
        print(f"  Tokens In/Out        : {data.get('vllm:prompt_tokens_total')} / {data.get('vllm:generation_tokens_total')}")
        print(f"  Total Iteration      : {data.get('vllm:iteration_tokens_total_sum')} tokens (sum)")
        
        print(f"\n Cache Stats ")
        print(f"  Tokens Cached        : {data.get('vllm:prompt_tokens_cached_total')}")
        print(f"  Tokens Recomputed    : {data.get('vllm:prompt_tokens_recomputed_total')}")
        print(f"  KV Cache Usage       : {float(data.get('vllm:kv_cache_usage_perc', 0)) * 100:.1f}%")

        print(f"\n Latency Performance ")
        print(f"  Time to 1st token    : {data.get('vllm:time_to_first_token_seconds_sum')} s (total sum)")
        print(f"  Inter-token latency  : {data.get('vllm:inter_token_latency_seconds_sum')} s (total sum)")
        print(f"  Time per Out Token   : {data.get('vllm:request_time_per_output_token_seconds_sum')} s")

        print(f"\n Hardware Stats (Per GPU)")
        print(f"  Estimated FLOPs      : {data.get('vllm:estimated_flops_per_gpu_total')}")
        print(f"  Estimated bytes read : {data.get('vllm:estimated_read_bytes_per_gpu_total')}")
        print(f"  Estimated bytes wrote: {data.get('vllm:estimated_write_bytes_per_gpu_total')}")

    else :
        print(f"FAILED : vLLM Metrics")



def monitor():
    try:
        timestamps = getServerTime()
        

        while True: 
            
            data = getLLMMetrics()
            os.system('clear')

            print(f"Metrics monitoring | Job : {jobId}")
            print(f"Started at {timestamps[0]} | Launched at {timestamps[1]}")

            displayGpu()

            if any(k.startswith('llamacpp:') for k in data):
                displayLlama(data)
            elif any(k.startswith('vllm:') for k in data):
                displayVLLM(data)
            else:
                print("\n Waiting for API")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n Fin du monitoring.")

        
monitor()