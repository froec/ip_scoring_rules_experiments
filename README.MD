# Code for the Paper "Scoring Rules and Calibration for Imprecise Probabilities"
Arxiv link

## Environment
Create a conda environment named "base" with python 3.11 and
 - matplotlib (3.8.4)
  - numpy (1.26.4)
  - pandas (2.2.2)
  - pytorch (2.3.0)
  - scikit-learn (1.4.2)
  - seaborn (0.13.2)
  - torchvision (0.18.1)
  - dill (0.3.7)
You can use environment.txt or environment_mimimal.yaml to set up the environemnt.

## Preparing the data
For framingham, download the dataset (a single csv file) from https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset?select=framingham.csv and put it in the data/ directory

For ACS, run "python download_acs.py" and then "python preprocess_acs.py" (run from within the data_scripts/ directory)

For CelebA:
1. Download the CelebA dataset from kaggle (https://www.kaggle.com/datasets/jessicali9530/celeba-dataset). The directory structure should look as data/celeba-kaggle/img_align_celeba/img_align_celeba/ 
2. Run "python compute_celeba_embeddings.py" (run from within the data_scripts/ directory)
OR
1. Simply download the computed embeddings from https://1drv.ms/u/c/4ff88d640b476479/EUqTT8bQqVZAsfbR5Ol8fAcBOvZQgIu0umxtsl7m6zB8BQ?e=bEcd4L and put them in the data/celeba_embeddings/ directory

## Running the experiments
To run the ACS experiment:
> python -m main --dataset acs --datadir data/ACS_unemployment_data/ --modeldir results_acs/ --logfile logs/acs_job.log \
		--batch_size 512 --outer_learning_rate_dro 0.1 --outer_epochs_dro 2000 --inner_learning_rate_dro 0.001 --inner_epochs_dro 500 \
		--epochs_individual_predictors 5000 --epochs_erm 5000\
		--c_params [0.1,0.3,0.7,0.9]

> python evaluation.py --modeldir results_acs/ --plotdir results_acs/ --c_params [0.1,0.3,0.7,0.9]

To run the CelebA experiment:
>python -m main --dataset celeba --datadir data/celeba_embeddings/ --modeldir results_celeba/ --logfile logs/celeba_job.log \
		--batch_size 512 --outer_learning_rate_dro 0.1 --outer_epochs_dro 2000 --inner_learning_rate_dro 0.001 --inner_epochs_dro 500 \
		--epochs_individual_predictors 5000 --epochs_erm 5000\
		--c_params [0.1,0.3,0.7,0.9]

>python evaluation.py --modeldir results_celeba/ --plotdir results_celeba/ --c_params [0.1,0.3,0.7,0.9]

To run the framingham experiment:
>python -m main --dataset framingham --datadir data/ --modeldir results_framingham/ --logfile logs/framingham_job.log \
		--outer_learning_rate_dro 0.1 --outer_epochs_dro 2000 --inner_learning_rate_dro 0.001 --inner_epochs_dro 500 \
		--epochs_individual_predictors 5000 --epochs_erm 5000\
		--c_params [0.1,0.3,0.7,0.9]\
		--no-minibatches

## Creating plots
Finally, to create the plots:
> python create_plots.py --plotdir results_acs/

> python create_plots.py --plotdir results_framingham/

> python create_plots.py --plotdir results_celeba/

To create helper plots for the paper, run plots/plots.py (see the instructions in plots.py)