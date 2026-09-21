#SBATCH --output=%x-%j.out 
#SBATCH --nodes=1
#SBATCH --partition=ampere_gpu
#SBATCH --cpus-per-gpu=12
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=Arthur.Van.Beersel@vub.be
#SBATCH --mail-type=ALL

cd $SLURM_SUBMIT_DIR

########################################
# MODULES
########################################

# module load ollama/0.24.0-GCCcore-14.2.0-CUDA-12.8.0
# module load Python/3.13.1-GCCcore-14.2.0

# module load SciPy-bundle/2025.06-gfbf-2025a
# module load openpyxl/3.1.5-GCCcore-14.2.0
# module load scikit-learn/1.7.0-gfbf-2025a
# module load matplotlib/3.10.3-gfbf-2025a

########################################
# ENVIRONMENT
########################################

source /data/brussel/vo/000/bvo00018/vsc11306/venv/cross-modal/bin/activate

export OLLAMA_MODELS=/data/brussel/vo/000/bvo00018/vsc11306/ollama
export OLLAMA_HOST=127.0.0.1:11434
export SERVER_URL=http://127.0.0.1:11434/
export OLLAMA_NUM_GPU=999
export OLLAMA_KEEP_ALIVE=-1

########################################
# START OLLAMA SERVER
########################################

echo "Starting Ollama server..."
export OLLAMA_LOGS=ollama_logs
ollama serve > $OLLAMA_LOGS/ollama_qwen.log 2>&1 &


########################################
# WAIT FOR SERVER
########################################

echo "Waiting for Ollama server..."

sleep 15
ollama pull gemma4:26b
ollama pull nomic-embed-text
OLLAMA_PID=$!
########################################
# DEBUG INFO
########################################

echo "Node:"
hostname

echo "Available models:"
ollama list

echo "Testing API:"
curl http://127.0.0.1:11434/api/tags

echo "GPU info:"
nvidia-smi

echo "Testing CUDA visibility:"
python -c "import torch; print(torch.cuda.is_available())"

########################################
# RUN PYTHON CODE
########################################

python -m Main_Scripts.main \
        --model 'gemma4:26b' \
        --datasets WikiSQL Adhesive_md Adhesive_json Adhesive_SMP \
        --rounds 5 \
        --sim_comp \
        --ollama_workers 10
########################################
# CLEANUP
########################################

kill $OLLAMA_PID