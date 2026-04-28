## How to use this project ?

### 1. Installation

Clone the repository to your local directory on the cluster:

    git clone https://github.com/gameking773/LLM_Monitoring.git
    cd <project-directory>

### 2. Configuration & Setup

The project uses a .env file to manage private paths and Slurm settings without hardcoding them.

Go into the cloned repo and create your environment file:

    cp .env.template .env

Edit .env with your specific credentials.

You need to make run.sh executable :

    chmod +x ./run.sh


Then, wait for the file to compile (it can take up to 30 minute).

Run the setup script. This will compile the engine for ARM/GPU (GH200) and download your model.

    ./run.sh sbatch setup/setupLlama.sh

*Compilation can take a long time and the job will automatically stops after 1h30, don't hesitate to increase it.*

## 3. Launch your server

Once the setup is complete, your model is stored in the /models directory. If you want to change it, simply download the new model and put it there. Then change the infos in the .env and voilà.

To start the Llama server:*

    ./run.sh sbatch runLlamaServer.sh

The server will start on the allocated node. To access the web interface :

- Identify the node name where the job is running with squeue --me

- Create an SSH tunnel from your local machine: ssh -L 8080:node-01:8080 <login>@romeo1.univ-reims.fr.

- Open http://localhost:8080 in your browser.

## 4. Monitor your LLM

To track your server's performance (VRAM usage, tokens/sec, etc.), launch the monitoring dashboard. It automatically detects your active Slurm job:

    python3 metrics.py


## What does the dashboard track ?

### 1. Sources

The dashboard uses multiple sources to track the server :

For the GPU, it uses the nvidia-smi command.

For Llama server, it uses the endpoints /metrics and /v1/models