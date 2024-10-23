#!/bin/bash
#SBATCH --job-name=framingham_job      # Job name
#SBATCH --output=results_framingham/framingham_output.txt # Standard output and error log
#SBATCH --ntasks=1                     # Number of tasks (processes)
#SBATCH --cpus-per-task=1              # Number of CPU cores per task
#SBATCH --time=0-24:00# Wall time
#SBATCH --mem=16000MB                    # Memory (in megabytes)
#SBATCH --partition=cpu-galvani

# Load Python module if necessary (depends on your cluster setup)
source ~/.bashrc
conda activate $WORK/.conda/py-311-pytorch

# Execute the Python script
python -m main --dataset framingham --datadir data/ --modeldir results_framingham/ --logfile logs/framingham_job.log \
		--outer_learning_rate_dro 0.1 --outer_epochs_dro 2000 --inner_learning_rate_dro 0.001 --inner_epochs_dro 500 \
		--epochs_individual_predictors 5000 --epochs_erm 5000\
		--c_params [0.1,0.3,0.7,0.9]\
		--no-minibatches

python evaluation.py --modeldir results_framingham/ --plotdir results_framingham/ --c_params [0.1,0.3,0.7,0.9]

conda deactivate