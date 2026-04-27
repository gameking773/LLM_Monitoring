## How to use this project ?

### 1. Clone the repository

### 2. Setup Llama.cpp

To use Llama.cpp, you will need to compile it first. Use the .env.template file to make your own .env. Then you can launch "run.sh" with this command :

    ./run.sh setup/setupLlama.sh

Then, wait for the file to compile (it can take up to 30 minute).

## 3. Launch your server

When you setup Llama, the model you picked in your .env will be downloaded. If you want to change it, simply download the new model and put it in the /models directory. Then change the infos in the .env and voilà.

Now that you have a model, you can launch Llama-server. 
Simply run this command :

    ./run.sh runLlamaServer.sh

