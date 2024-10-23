# run this using python -m main, in order for imports/dill to work
# use python -m main --no-minibatches for using full data instead of minibatches, this will set the args.minibatches to false
# for framingham: python -m main --dataset framingham --logfile results_framingham_loglossbase/output.log --datadir data/ --no-minibatches --epochs_individual_predictors 5000 --c_params [0.1,0.2] --modeldir results_framingham/
# for ACS: python -m main --dataset acs --datadir data/ACS_unemployment_data/
# for celebA: python -m main --dataset celeba --datadir data/celeba_embeddings/ --batch_size 512 --modeldir results_celeba/ --logfile results_celeba/celeba_job.log --outer_learning_rate_dro 1.0 --outer_epochs_dro 250 --inner_learning_rate_dro 0.001 --c_params [0.1,0.2,0.3,0.7,0.8,0.9]

# current test:
# python -m main --dataset celeba --datadir data/celeba_embeddings/ --batch_size 256 --epochs_individual_predictors 2000 --epochs_erm 2000 --modeldir results_celeba/ --logfile results_celeba/celeba_job.log --outer_learning_rate_dro 1.0 --outer_epochs_dro 250 --inner_learning_rate_dro 0.001 --c_params [0.2]

# to do the AK-CA pair:
# python -m main --dataset acs --datadir data/ACS_Unemployment_data/ --states ['AK','CA'] -- --outer_learning_rate_dro 0.1 --outer_epochs_dro 500 --inner_epochs_dro 500 --inner_learning_rate_dro 0.001 --c_params [0.2,0.8] --epochs_individual_predictors 5000 --epochs_erm 5000
# and use a subsample of [5058,50000]

# to draw entropy comparison curve:
# python -m main --dataset acs --datadir data/ACS_unemployment_data/ --plot_binary_entropy_curves --superquick --states ['AK','CA']

# L-BFGS test: python -m main.py --dataset framingham --no-minibatches --epochs_individual_predictors 100

# python ML/data science stack
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import torch


# sklearn imports
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_curve, confusion_matrix


# general python
import os
import copy
import itertools
import dill
import argparse
import logging
import ast

# our modules
from data_scripts.load_framingham import FraminghamLoader
from data_scripts.load_acs import ACSLoader
from data_scripts.load_celeba import CelebALoader
from group_data_loader import GroupDataLoader, FullGroupDataLoader, MiniBatchGroupDataLoader, TrivialDataLoader
from trainers import BinaryReg_Trainer, BinaryReg_ExpGradDRO_Trainer
from predictors import PrecisePredictor, MinMaxEnsemble

from utils import printdict_4f, computeAccuracy, SimpleLogger
from loss_functions import CostSensitiveLoss, CostSensitiveLossWithRejectOption, \
                            BrierLoss, LogLoss, SphericalScore, AsymmetricScore


seed = 42 # random seed
np.random.seed(seed)
torch.manual_seed(seed)


# python -m main --batch_size=256 --learning_rate_individual_predictor 0.001 
def parse_args():
    parser = argparse.ArgumentParser(description="Parse arguments for the model.")
    parser.add_argument("--logfile", type=str, default="output.log", help="Log file to write to.")
    parser.add_argument("--dataset", type=str, default="framingham", choices=["framingham","acs","celeba"], help="Name of the dataset to use.")
    parser.add_argument("--datadir", type=str, default="data/", help="Data directory.")
    parser.add_argument("--modeldir", type=str, default="models/", help="Directory for saving trained models.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of data used for testing (group-wise).")

    # batch_size = 256 is often a reasonable batch_size with logistic regression
    parser.add_argument("--batch_size", type=int, default=64, help="Size of mini-batches, or None for using full data.")
    parser.add_argument("--minibatches", default="True", action="store_true")
    parser.add_argument("--no-minibatches", dest="minibatches", action="store_false", \
                        help="Use mini-batches for training. If false, then batch_size will be set to None.")

    # Example value for c_params: [0.1,0.2,0.3,0.4] ; important: don't use spaces!
    parser.add_argument("--c_params", type=str, default='[0.1,0.9]', help="Parameters c for cost-sensitive losses, which are used for training some models (list of floats; no spaces!)")


    # individual predictor parameters
    # ###### NOTE: I previously used lr=0.01 when passing the full data...but with mini-batches, need smaller learning rate
    parser.add_argument("--learning_rate_individual_predictor", type=float, default=0.001,\
     help="Learning rate for the individual predictors, which are trained only on the data of a single groups.")

    parser.add_argument("--epochs_individual_predictors", type=int, default=5000,\
     help="Number of epochs for the individual predictors, which are trained only on the data of a single groups.")

    # relates to the sklearn individual predictor
    parser.add_argument("--plot_sklearn_roc", action="store_true", default=False, help="Plot the ROC curve for the trained sklearn model.")

    # parameters for ERM generalist models
    parser.add_argument("--learning_rate_erm", type=float, default=0.001,\
     help="Learning rate for ERM (trained on data of all groups).")
    parser.add_argument("--epochs_erm", type=int, default=10000,\
     help="Epochs for ERM (trained on data of all groups).")

    # parameters for DRO models
    parser.add_argument("--outer_learning_rate_dro", type=float, default=2.,\
     help="Outer learning rate for training DRO models.")

    parser.add_argument("--inner_learning_rate_dro", type=float, default=0.01,\
     help="Inner learning rate for training DRO models.")

    parser.add_argument("--outer_epochs_dro", type=int, default=100,\
     help="Number of outer epochs for training DRO models.")
    parser.add_argument("--inner_epochs_dro", type=int, default=250,\
     help="Number of inner epochs for training DRO models.")

    parser.add_argument("--superquick", action="store_true", default=False,\
     help="Ignore all other epoch parameters and perform a super quick training for testing purposes")

    # only to be used in combination with ACS dataset
    parser.add_argument("--states", type=str, default="", \
        help="If you want to use only ACS data for specific states, enter their state codes here as a list of the form ['AK','CA']")

    return parser.parse_args()

args = parse_args()


# set up our logger
logger = SimpleLogger('logger', log_file=args.logfile, level=logging.DEBUG)

# parse the c_params as a list of float
args.c_params = ast.literal_eval(args.c_params)

logger.debug(args)


# select the dataset to use
if args.dataset == "framingham":
    data_dictionary_loader = FraminghamLoader(args.datadir, logger, subsample=None, seed=seed, test_size=args.test_size)

elif args.dataset == "acs":
    # parse the states unless we want to use all states
    if args.states == "":
        # we will use all states
        args.states = None 
    else:
        args.states = ast.literal_eval(args.states) # [5058,50000]
    data_dictionary_loader = ACSLoader(args.datadir, logger,subsample=None, seed=seed, test_size=args.test_size, states=args.states)
 

elif args.dataset == "celeba":
    data_dictionary_loader = CelebALoader(args.datadir, logger, seed=seed) # test_size not supported for celebA

data_dictionary = data_dictionary_loader.data_dictionary
states = data_dictionary_loader.states
feature_names = data_dictionary_loader.feature_names

# is superquick training enabled? for testing purposes
if args.superquick:
    args.epochs_individual_predictors = 10
    args.outer_epochs_dro = 1
    args.inner_epochs_dro = 10
    args.epochs_erm = 10

if args.minibatches == False:
    logger.debug("Using full data instead of minibatches")
    args.batch_size = None


# try to create modeldir where we save the trained models
if not os.path.exists(args.modeldir):
    os.makedirs(args.modeldir)




######################################################
"""
Model Training
We differentiate between
1. Individual models (trained on the data of each state (group) separately)
2. The generalist ERM model (trained on the data of all states combined)
3. THe DRO model
"""
######################################################


# in this dictionary, we save all the trainers (each of which has its underlying model)
# so that we can later evaluate them
all_trainers = {}
accuracies_sklearn = []


####################
# First, we train a simple logistic regressor for each state (group) separately
####################
for state in states:
    logger.debug("State: %s" % state)
    state_dict = data_dictionary[state]
    X_train = state_dict["X_train"]
    y_train = state_dict["y_train"]
    X_test = state_dict["X_test"]
    y_test = state_dict["y_test"]

    logger.debug("N_train: %s" % len(X_train))
    logger.debug("N_test: %s" % len(X_test))
    logger.debug("positive train labels: %.4f" % (y_train.sum()))
    logger.debug("positive test labels: %.4f" % (y_test.sum()))

    surr_loss = torch.nn.BCELoss(reduction='none')
    logreg_trainer = BinaryReg_Trainer(X_train.shape[1], lr=args.learning_rate_individual_predictor, epochs=args.epochs_individual_predictors, logger=logger)

    # fit and apply a standard scaler
    logreg_trainer.scaler = StandardScaler()
    X_train_trans = logreg_trainer.scaler.fit_transform(X_train)
    def f(X_test, scaler=logreg_trainer.scaler):
        return scaler.transform(X_test)
    logreg_trainer.transform = f


    if args.batch_size is None:
        # we use the full data
        logger.debug("training individual predictor with full data")
        trivial_data_loader = TrivialDataLoader(X_train_trans, y_train)
        logreg_trainer.train(dataloader=trivial_data_loader, criterion=surr_loss, verbose=True)
        

    else:
        # or with mini-batches
        logger.debug("training individual predictor with mini-batches")
        minibatch_data_loader = MiniBatchGroupDataLoader(X_train_trans, y_train, group_memberships=np.zeros(len(y_train)), n_groups=1,\
                                 batch_size=args.batch_size, verbose=True)
        logreg_trainer.train(dataloader=minibatch_data_loader, criterion=surr_loss, verbose=True)


    bayes_risk = logreg_trainer.finalTrainingLoss
    logger.debug("Bayes risk: %.6f" % bayes_risk)
    logger.debug(logreg_trainer.model.layer1[0].weight.flatten()[:20])
    X_test_trans = logreg_trainer.scaler.transform(X_test)
    acc = logreg_trainer.computeAccuracy(X_test_trans,y_test)
    logger.debug("accuracy: %s" % acc)
    logreg_preds = logreg_trainer.predict(X_test_trans)
    logger.debug("fraction of prediction=1: %.4f" % ((logreg_preds>0.5).sum()/len(logreg_preds)))
    for c_param in args.c_params:
        logger.debug("when using cost-sensitive loss with c=%s, fraction of prediction=1: %.4f" % (c_param, (logreg_preds>c_param).sum()/len(logreg_preds)))
    base_rate = ((y_train==1.).sum()/len(y_train))
    logger.debug("train base rate of y=1: %.4f" % base_rate)
    logger.debug("accuracy of train base rate predictor: %.4f" % ((y_test==(0. if base_rate < 0.5 else 1.)).sum()/len(y_test)))


    # show the most important features and their coefficients
    logreg_coefs = list(logreg_trainer.model.layer1[0].weight.flatten())
    logreg_coefs_zipped = zip(feature_names,logreg_coefs)
    sorted_coefs = pd.DataFrame(sorted(logreg_coefs_zipped, key=lambda item: abs(item[1]), reverse=True))
    logger.debug("top sorted (by absolute value) coefficients:")
    logger.debug(sorted_coefs[:3])


    # save the trained models
    all_trainers['model-logreg-%s'%state] = logreg_trainer
    # also with accuracy
    all_trainers['model-logreg-%s-accuracy'%state] = acc
    # and final training loss
    all_trainers[('model-logreg-%s'%state)+'-final-training-loss'] = bayes_risk 

    

    # sanity check using sklearn
    # do the logistic regression with sklearn
    logreg = LogisticRegression(random_state=seed, max_iter=5000)
    scaler = StandardScaler()
    X_train_trans = scaler.fit_transform(X_train)
    X_test_trans = scaler.transform(X_test)
    logreg.fit(X_train_trans, y_train)
    acc_sk = logreg.score(X_test_trans,y_test)
    logreg_sklearn_preds = logreg.predict_proba(X_test_trans)[:,1].flatten()
    logger.debug("sklearn Logistic Regression accuracy on test set: %.4f" % acc_sk)
    logger.debug("sklearn : fraction of prediction=employed: %.4f" % ((logreg_sklearn_preds>0.5).sum()/len(logreg_sklearn_preds)))
    accuracies_sklearn.append(acc_sk)
    logger.debug("coefficients:")
    #logger.debug(logreg.coef_.flatten()[:20])
    #logger.debug(logreg.intercept_)
    all_trainers['model-logreg-%s-sklearn-coef_'%state] = logreg.coef_.flatten()
    logger.debug("confusion_matrix of sklearn model:")
    tn, fp, fn, tp = confusion_matrix(y_test, logreg.predict(X_test_trans)).ravel()
    logger.debug("(tn, fp, fn, tp)")
    logger.debug((tn, fp, fn, tp))


    if args.plot_sklearn_roc:
        fpr, tpr, thresholds = roc_curve(y_test, logreg_sklearn_preds)
        fpr_base, tpr_base, thresholds_base = roc_curve(y_test, base_rate*torch.ones(len(y_test)))
        plt.plot(fpr,tpr)
        plt.plot(fpr_base,tpr_base,c='k')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.title('ROC curve for Heart disease classifier')
        plt.xlabel('False positive rate (1-Specificity)')
        plt.ylabel('True positive rate (Sensitivity)')
        plt.grid(True)
        plt.show()
    


    # compare to the log-loss of the naive classifier which simply predicts the base rate
    criterion=torch.nn.BCELoss(reduction='none')
    y_preds_naive = np.mean(y_train)*np.ones(len(y_train))
    naiveloss = torch.mean(criterion(torch.tensor(y_preds_naive),torch.tensor(y_train)).flatten())
    logger.debug("naive (constant base rate predictor) training loss: %s" % naiveloss)


    logger.debug("");logger.debug("");


# logger.debug all the bayes risks of individual predictors
logger.debug("Bayes risks of the states (empirical risk of optimal individual predictor on that state):")
brs = {s : all_trainers[('model-logreg-%s'%s)+'-final-training-loss'] for s in states}
sorted_brs_dict = dict(sorted(brs.items(), key=lambda item: item[1], reverse=True))
logger.debug_dict_4f(sorted_brs_dict)
logger.debug("");
logger.debug("");





####################
# compare performance of individual predictors evaluated on other states
# e.g. for ACS data, taking the alaska predictor and evaluating it on hawaii

if len(states)<3:
    pairs = list(itertools.product(range(len(states)), range(len(states))))
    dists = []
    for i,j in pairs:
        logger.debug("Using individual predictor for %s on data of %s" % (states[i],states[j]))
        model_i = all_trainers['model-logreg-%s'%states[i]]
        X_test_j = data_dictionary[states[j]]["X_test"]
        y_test_j = data_dictionary[states[j]]["y_test"]
        X_test_j = model_i.transform(X_test_j)
        logger.debug("accuracy: %.4f" % model_i.computeAccuracy(X_test_j,y_test_j))
        mean_train_loss_j = np.mean(model_i.computeLossDistribution(X_test_j,y_test_j))
        logger.debug("mean train loss: %.4f" % mean_train_loss_j)
        logger.debug("");




####################
# ERM GENERALIST
# Next, we train a simple logistic regressor on the data of all states simultaneously
# 1. with logloss (standard log reg)
# 2. with asymmetric loss
####################
X_train_combined = np.vstack([data_dictionary[s]["X_train"] for s in states])
y_train_combined = np.hstack([data_dictionary[s]["y_train"] for s in states])
# also collect all of the test data together
X_test_combined = np.vstack([data_dictionary[s]["X_test"] for s in states])
y_test_combined = np.hstack([data_dictionary[s]["y_test"] for s in states])
logger.debug((X_train_combined.shape,y_train_combined.shape))

# save in data dictionary
data_dictionary["X_train_combined"] = X_train_combined
data_dictionary["y_train_combined"] = y_train_combined
data_dictionary["X_test_combined"] = X_test_combined
data_dictionary["y_test_combined"] = y_test_combined

# now apply a scaler, we will then save this scaler with every one the trainers which is trained on combined data
combined_scaler = StandardScaler()
X_train_trans_combined = combined_scaler.fit_transform(X_train_combined)
X_test_trans_combined = combined_scaler.transform(X_test_combined)
def apply_combined_scaler(data,scaler=combined_scaler):
    return scaler.transform(data)


# first a quick sklearn comparison
logreg = LogisticRegression(random_state=seed, max_iter=5000)
logreg.fit(X_train_trans_combined, y_train_combined)
acc_sk = logreg.score(X_test_trans_combined,y_test_combined)
logger.debug("Sklearn logistic regression on combined data accuracy: %.4f" % acc_sk)


erm_surrogate_losses = dro_surrogate_losses = [('log', torch.nn.BCELoss(reduction='none'))] + [('asymm(c=%s)'%c_param, AsymmetricScore(c=c_param)) for c_param in args.c_params]
# we require the log loss for later initialization
assert 'log' in [l[0] for l in erm_surrogate_losses]

# we use the ERM trained with log loss for later initializing the DRO models, since this provides a good initialization
logloss_trainer = None 
for surr_loss_name, surr_loss in erm_surrogate_losses:             
    logger.debug("training ERM generalist with surrogate loss: %s" % surr_loss_name)

    logreg_trainer_combined = BinaryReg_Trainer(X_train_combined.shape[1], lr=args.learning_rate_erm, epochs=args.epochs_erm, logger=logger)
    # preprocessing: scaling
    logreg_trainer_combined.transform = apply_combined_scaler
    

    if args.batch_size is None:
        # use full data
        trivial_data_loader = TrivialDataLoader(X_train_trans_combined, y_train_combined)
        logreg_trainer_combined.train(verbose=True, criterion=surr_loss, dataloader=trivial_data_loader)
    else:
        mini_data_loader = MiniBatchGroupDataLoader(X_train_trans_combined, y_train_combined, \
                        group_memberships=np.zeros(len(X_train_trans_combined)), n_groups=1, batch_size=args.batch_size, verbose=True)
        logreg_trainer_combined.train(verbose=True, criterion=surr_loss, dataloader=mini_data_loader)


    logger.debug("coefficients of first 20 final weights:")
    logger.debug(logreg_trainer_combined.model.layer1[0].weight.flatten()[:20])

    logreg_preds_combined = logreg_trainer_combined.predict(X_test_combined)
    acc_combined = logreg_trainer_combined.computeAccuracy(X_test_trans_combined,y_test_combined)
    logger.debug("our accuracy on all test data of logreg trained on combined data: %.4f" % acc_combined)
    logger.debug("");
    all_trainers['model-combined-%s-ypreds'%surr_loss_name] = logreg_preds_combined
    all_trainers['model-combined-%s'%surr_loss_name] = logreg_trainer_combined

    if surr_loss_name == 'log':
        logloss_trainer = logreg_trainer_combined





####################
# DRO (Logistic) Regression
####################
dro_surrogate_losses = [('log', torch.nn.BCELoss(reduction='none'))] + [('asymm(c=%s)'%c_param, AsymmetricScore(c=c_param)) for c_param in args.c_params]


for surr_loss_name, surr_loss in dro_surrogate_losses:            
    logger.debug("Training Exponentiated Gradient Descent DRO Model..")
    logger.debug("using surrogate loss function: %s" % surr_loss_name)
    #### learning rates depends crucially on the data set
    dro_trainer_combined = BinaryReg_ExpGradDRO_Trainer(X_train_combined.shape[1], \
                            outer_lr=args.outer_learning_rate_dro, inner_lr=args.inner_learning_rate_dro, logger=logger) 
    # for the DRO model, it is important to use a good initialization, otherwise training can be extremely slow
    # we therefore use the ERM model as a start
    # we use the ERM trained with log loss here!
    dro_trainer_combined.model = copy.deepcopy(logloss_trainer.model)
    groups = np.hstack([i*np.ones(len(data_dictionary[s]["y_train"])) for i,s in enumerate(states)])

    # train using the respective surrogate loss function

    if args.batch_size is None:
        logger.debug("training DRO with full data")
        dro_group_data_loader = FullGroupDataLoader(X_train_trans_combined,y_train_combined, groups, len(states))
    else:
        logger.debug("training DRO with mini-batches")
        dro_group_data_loader = MiniBatchGroupDataLoader(X_train_trans_combined,y_train_combined, groups, len(states), args.batch_size)

    # we also can provide test data, so that every couple epochs we evaluate the DRO loss on this test data
    # in this case, the "test" data coincides with the training data (!) [sic]; and we use the group knowledge in evaluation
    # we want to know the DRO loss on train data..
    dro_trainer_combined.train(dataloader=dro_group_data_loader, X_eval=X_train_trans_combined, y_eval=y_train_combined,\
                                group_memberships_eval=groups, n_groups=len(states), criterion=surr_loss, \
                                verbose=False, outer_epochs=args.outer_epochs_dro, inner_epochs=args.inner_epochs_dro)
    dro_trainer_combined.transform = apply_combined_scaler
    logger.debug("final group weightings: ")
    final_weights = dict(zip(states,dro_trainer_combined.final_lam))
    logger.debug_dict_4f(final_weights)

    dro_acc_combined = dro_trainer_combined.computeAccuracy(X_test_trans_combined,y_test_combined)
    logger.debug("DRO model accuracy on combined test data: %.4f" % dro_acc_combined)
    logger.debug("");logger.debug("");
    all_trainers['model-%s-dro-ypreds'%surr_loss_name] = dro_trainer_combined.predict(X_test_combined)
    all_trainers['model-%s-dro'%surr_loss_name] = dro_trainer_combined


logger.debug("training of all DRO models finished")





######################################################
"""
Trainers->Predictors
For each trainer which works on combined data, construct a predictor
and also add the GBR, which is based on an ensemble
"""
######################################################
# these will be evaluated later, together with the GBR
precise_trainers = ['model-combined-log'] + ['model-combined-asymm(c=%s)'%c_param for c_param in args.c_params] \
                    + [('model-%s-dro'%s[0]) for s in dro_surrogate_losses]#, 'model-naive-logreg-dro']
precise_predictors = []
for m in precise_trainers:
    precise_predictors.append(PrecisePredictor(all_trainers[m],m))


####################
# GBR : generalized bayes rule from individual predictors
####################
GBREnsemble = MinMaxEnsemble([PrecisePredictor(all_trainers['model-logreg-%s'%s],'model-logreg-%s'%s) for s in states], name='GBR')



######################################################
"""
Now save all results, evaluation takes place in another script
"""
######################################################

predictors_to_evaluate = precise_predictors + [GBREnsemble] #['model-asymm-combined']

dill.dump(data_dictionary_loader, open(args.modeldir + "data_dictionary_loader.pkl","wb"))
dill.dump(predictors_to_evaluate, open(args.modeldir + "predictors_to_evaluate.pkl","wb"))