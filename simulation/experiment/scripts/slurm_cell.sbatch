#!/bin/bash
#SBATCH --job-name=cr-1on1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-14%8
# One array task = one persona x severity cell (generation, then judging), with a private Ollama server per job.
# Edit the paths below for your cluster; the paper used one 24-48 GB GPU per job and Ollama 0.32.9.
LEVELS=(110 111 112 240 241 242 250 251 252 260 261 262 270 271 272)
HERE=${HERE:?set HERE to simulation/experiment}
export OLLAMA_HOST=127.0.0.1:$((11700 + SLURM_JOB_ID % 200))
ollama serve > "$HERE/results/serve_${SLURM_JOB_ID}.log" 2>&1 &
sleep 10
export OLLAMA_URL=http://$OLLAMA_HOST/v1
bash "$HERE/scripts/run_cell.sh" "${LEVELS[$SLURM_ARRAY_TASK_ID]}"
bash "$HERE/scripts/measure.sh" "$HERE/results/rerun" "${LEVELS[$SLURM_ARRAY_TASK_ID]}"
