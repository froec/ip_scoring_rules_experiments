#!/bin/bash
#SBATCH --job-name=celeba_job #Job name
#SBATCH --output=results_celeba/celeba_output.txt # Standard output and error log
#SBATCH --ntasks=1                     # Number of tasks (processes)
#SBATCH --cpus-per-task=1              # Number of CPU cores per task
#SBATCH --time=0-36:00# Wall time
#SBATCH --mem=36000MB                    # Memory (in megabytes)
#SBATCH --partition=cpu-galvani

# Load Python module if necessary (depends on your cluster setup)
source ~/.bashrc
conda activate /mnt/qb/work/williamson/wec701/.conda/py-311-pytorch

# Execute the Python script
python -m main --dataset celeba --datadir data/celeba_embeddings/ --modeldir results_celeba/ --logfile logs/celeba_job.log \
		--batch_size 512 --outer_learning_rate_dro 0.1 --outer_epochs_dro 2000 --inner_learning_rate_dro 0.001 --inner_epochs_dro 500 \
		--epochs_individual_predictors 5000 --epochs_erm 5000\
		--c_params [0.1,0.3,0.7,0.9]

python evaluation.py --modeldir results_celeba/ --plotdir results_celeba/ --c_params [0.1,0.3,0.7,0.9]

conda deactivate