import os
import dill
import logging
import argparse
import ast

import numpy as np
import pandas as pd

from utils import computeAccuracy, SimpleLogger
from loss_functions import CostSensitiveLoss, CostSensitiveLossWithRejectOption, \
    BrierLoss, LogLoss, SphericalScore, AsymmetricScore
from decision_calibration import compute_decision_calibration, compute_ip_decision_calibration

# Example usage: python evaluation.py --modeldir results_framingham/ --plotdir results_framingham/

print("evaluating...")
# set up our logger
logger = SimpleLogger('logger', log_file='evaluation.log', level=logging.DEBUG)

def parse_args():
    parser = argparse.ArgumentParser(description="Parse arguments for the model.")
    parser.add_argument("--modeldir", type=str, default="models/", help="Directory for the saved trained models.")
    parser.add_argument("--plotdir", type=str, default="results/", help="Directory for saving created plots/evaluation results.")
    parser.add_argument("--create_plots", default=False, action="store_true", help="Create plots (requires matplotlib)")
    parser.add_argument("--show_plots", type=int, default=1, help="Show the created plots. 0=don't show, 1=show most important, 2=show all")
    # Example value for c_params: [0.1,0.2,0.3,0.4] ; important: don't use spaces!
    parser.add_argument("--c_params", type=str, default='[0.1]', help="Parameters c for cost-sensitive losses, which are used for training some models (list of floats; no spaces!)")
    return parser.parse_args()

args = parse_args()

# parse the c_params as a list of float
args.c_params = ast.literal_eval(args.c_params)


if args.create_plots:
    logger.debug("we will create plots..")
    import matplotlib.pyplot as plt
    import seaborn as sns

logger.debug("loading dill files")
data_dictionary_loader = dill.load(open(args.modeldir + "data_dictionary_loader.pkl","rb"))
predictors_to_evaluate = dill.load(open(args.modeldir + "predictors_to_evaluate.pkl","rb"))
logger.debug("finished loading dill files")

data_dictionary = data_dictionary_loader.data_dictionary
states = data_dictionary_loader.states
feature_names = data_dictionary_loader.feature_names

X_train_combined = data_dictionary["X_train_combined"]
X_test_combined = data_dictionary["X_test_combined"]
y_train_combined = data_dictionary["y_train_combined"]
y_test_combined = data_dictionary["y_test_combined"]



########################################################################################################
########################################################################################################
# Evaluation of predictors in a loop in terms of loss / IP score, where each predictor is evaluated on each of the states
# first evaluate only the precise models
#############


# this list (later transformed to data frame) will hold the results of evaluation of all predictors on the combined data (of all states)
all_combined_res = []

# for the evaluation across states, we employ a single surrogate_loss (across all models)
# to compare the mean surrogate loss across states (i.e. look at its range)
surrogate_loss = LogLoss()
accuracy_loss = CostSensitiveLoss(c=0.5) # loss to be used when computing accuracy

logger.debug("")
logger.debug("#############")
logger.debug("Beginning evaluation of predictors")
logger.debug("#############")
logger.debug("")

ipscores = []

# loss functions to be used for evaluation
cost_sensitive_losses = [CostSensitiveLoss(c_param) for c_param in args.c_params] 
loss_funcs = cost_sensitive_losses #+ [LogLoss()]

# we might want to exclude some predictors since they are not too interesting
# otherwise the plots get too crowded
for predictor in predictors_to_evaluate[:]:
    if "model-combined-asymm" in predictor.name:
        predictors_to_evaluate.remove(predictor)

logger.debug("predictors to evaluate:")
logger.debug(predictors_to_evaluate)


for predictor in predictors_to_evaluate:
    dro_weights = {}
    if "model-asymm(c=" in predictor.name:
        print(predictor.name)
        final_weights = predictor.trainer.final_lam
        print(final_weights)
        dro_weights[predictor.name] = final_weights
        dill.dump(dro_weights, open(args.plotdir + "final_dro_weights.pkl","wb"))



for predictor in predictors_to_evaluate:

    # evaluate the predictor on combined data (all test data)
    logger.debug("evaluating predictor=%s" % predictor.name)
    logger.debug("-------------------")

    # evaluate on data of all states
    y_preds = predictor.getRecommendedActions(X_test_combined,accuracy_loss)
    overall_acc_model = computeAccuracy(y_preds,y_test_combined)
    logger.debug("the overall accuracy of the model is: %.4f" % overall_acc_model)

    preds = predictor.predict(X_test_combined)
    #print(preds)

    if predictor.name == "GBR":
        print("GBR:")
        print("ensemble preds:")
        print(np.array([mem.predict(X_test_combined) for mem in predictor.ensemble_members])) 
        print("recommended actions:")
        print(y_preds)
    

    # evaluate the model on each state
    diff_to_avgs = {}

    loss_train_dist_by_state = {} 
    loss_test_dist_by_state = {} 
    logger.debug("Now: state-wise evaluation:")
    for s in states:
        sdict = data_dictionary[s]
        # compute the accuracy on the given state
        class_preds = predictor.getRecommendedActions(sdict["X_test"],accuracy_loss)
        acc = computeAccuracy(class_preds,sdict["y_test"])

        data_dictionary[s]['accuracy_of_' + predictor.name] = acc
        diff_to_avgs[s] = acc-overall_acc_model

        #losses_train = all_trainers[modelname].computeLossDistribution(sdict["X_train"],sdict["y_train"])

        action_recs = predictor.getRecommendedActions(sdict["X_train"],surrogate_loss)
        losses_train = surrogate_loss.ell(action_recs, sdict["y_train"])
        loss_train = np.mean(losses_train) # this is the mean training surrogate loss for this state
        loss_train_dist_by_state[s] = loss_train

        #losses_test = all_trainers[modelname].computeLossDistribution(sdict["X_test"],sdict["y_test"])
        losses_test = surrogate_loss.ell(predictor.getRecommendedActions(sdict["X_test"],surrogate_loss), sdict["y_test"])
        loss_test = np.mean(losses_test) # this is the mean test surrogate loss for this state
        loss_test_dist_by_state[s] = loss_test



    # now look at results across states
    statewise_accs_of_model = [data_dictionary[s]['accuracy_of_' + predictor.name] for s in states]
    #logger.debugdict_4f(dict(zip(states,statewise_accs_of_model)))

    
    if args.create_plots:
        plt.figure()
        plt.title("State-wise accuracies of predictor="+predictor.name)
        plt.hist(statewise_accs_of_model,bins=len(statewise_accs_of_model))
        plt.axvline(x=overall_acc_model, color='r', linestyle='--')
        plt.savefig(args.plotdir + ("statewise_accuracies_of_%s.pdf"%predictor.name), bbox_inches='tight')
        if args.show_plots >= 2.:
            plt.show()
        else:
            plt.close()

    """
    # see on which states it does better and on which worse
    logger.debug("Compute 'accuracy of model on this state' - 'accuracy of model on combined data'")
    diff_to_avgs = dict(sorted(diff_to_avgs.items(), key=lambda item: item[1], reverse=True))
    logger.debugdict_4f(diff_to_avgs)
    """

    # look at the range of the accuracies of the model across all states
    logger.debug("the range of accuracies: max accuracy - min accuracy = ")
    maxmindiff_acc = max(statewise_accs_of_model)-min(statewise_accs_of_model)
    logger.debug("%.4f" % maxmindiff_acc)


    
    # look at the trainining loss distribution over states
    # for each state, we know the mean training surrogate loss (log loss)
    if args.create_plots:
        plt.figure()
        plt.title("Training loss distribution over states")
        plt.hist(loss_train_dist_by_state.values(),bins=len(states))
        plt.savefig(args.plotdir + ("training_loss_dist_by_state_of_%s.pdf"%predictor.name), bbox_inches='tight')
        if args.show_plots >= 2.:
            plt.show()
        else:
            plt.close()
    logger.debug("max/min surrogate training loss (mean loss) over states:")
    min_ = min(loss_train_dist_by_state.values())
    max_ = max(loss_train_dist_by_state.values())
    logger.debug("%.4f - %.4f = %.4f" % (max_, min_, max_-min_))

    
    # the same for test loss
    if args.create_plots:
        plt.figure()
        plt.title("Test surrogate loss distribution over states")
        plt.hist(loss_test_dist_by_state.values(),bins=len(states))
        plt.savefig(args.plotdir + ("test_loss_dist_by_state_of_%s.pdf"%predictor.name), bbox_inches='tight')
        if args.show_plots >= 2.:
            plt.show()
        else:
            plt.close()
    logger.debug("max/min surrogate test loss (mean loss) over states:")
    min_ = min(loss_test_dist_by_state.values())
    max_ = max(loss_test_dist_by_state.values())
    logger.debug("%.4f - %.4f = %.4f" % (max_, min_, max_-min_))


    # for the currently fixed predictor,
    # compute the IP score for different loss functions, for both training and test data (state-wise and then take max)
    # meaning the data model represents the set of states, each state is viewed as a probability measure
    # (either its empirical train or its empirical test distribution)
    # and we compute the upper expectation of the IP scoring rule under this data model
    # each loss function corresponds to an IP scoring rule
    ipscores_ofthispredictor = []
    logger.debug("IP Scores under the IP data model:")
    for l in loss_funcs:
        max_meanloss_train = (-99999,None)
        max_meanloss_test = (-99999,None)
        for s in states:
            sdict = data_dictionary[s]
            a_stars_train = predictor.getRecommendedActions(sdict["X_train"], l)
            meanloss_train = np.mean(l.ell(a_stars_train, sdict["y_train"]))
            if meanloss_train>max_meanloss_train[0]:
                max_meanloss_train = (meanloss_train,s)

            a_stars_test = predictor.getRecommendedActions(sdict["X_test"], l)
            meanloss_test = np.mean(l.ell(a_stars_test, sdict["y_test"]))
            if meanloss_test>max_meanloss_test[0]:
                max_meanloss_test = (meanloss_test,s)

        ipscores.append((predictor.name, l.name, max_meanloss_train[0], max_meanloss_test[0]))
        ipscores_ofthispredictor.append((l.name, max_meanloss_train[0], max_meanloss_test[0]))


    ipscores_ofthispredictor = pd.DataFrame(ipscores_ofthispredictor, columns=['Loss Name', 'IP Score (Train)', 'IP Score (Test)'])
    logger.debug(ipscores_ofthispredictor)
    logger.debug("");logger.debug("");



    # now evaluate expteced loss in the standard ML way, using the combined data set (precise data model)
    # now evaluate different loss functions, both on training and on test set for COMBINED data (of all states)
    combined_reslist = []
    logger.debug("Now evaluation on combined data (of all states)")
    logger.debug("Loss function: mean train loss / mean test loss")
    for l in loss_funcs:
        # now the recommended actions of course depend on the loss function!
        a_stars_train = predictor.getRecommendedActions(X_train_combined, l)
        a_stars_test = predictor.getRecommendedActions(X_test_combined, l)
        meanloss_train = np.mean(l.ell(a_stars_train, y_train_combined))
        meanloss_test = np.mean(l.ell(a_stars_test, y_test_combined))
        combined_reslist.append((l.name, meanloss_train, meanloss_test))
        all_combined_res.append((predictor.name,l.name, meanloss_train, meanloss_test))


    combined_resdf = pd.DataFrame(combined_reslist,columns=['Loss Name','Train Loss','Test Loss'])
    logger.debug(combined_resdf)
    logger.debug("");logger.debug("");



# now put all the results together (for all models): IP scores with respect to IP data model
ipscores_df = pd.DataFrame(ipscores,columns=['Predictor Name','Loss Name', 'IP Score (Train)', 'IP Score (Test)'])

# now put all the results together (for all models): (potentially imprecise) predictors with respect to precise data model
all_combined_res_df = pd.DataFrame(all_combined_res,columns=['Predictor Name','Loss Name', 'Train Loss','Test Loss'])


# which predictor has optimal IP train score under which loss?
logger.debug("optimal predictors for IP scores under the different losses:")
best_predictors = ipscores_df.groupby('Loss Name')['IP Score (Train)'].idxmin()
# Use these indices to get the corresponding predictor names
result = ipscores_df.loc[best_predictors, ['Loss Name', 'Predictor Name', 'IP Score (Train)']]
# Sort the result by Loss Name for better readability
result = result.sort_values('Loss Name').reset_index(drop=True)

# Display the result
logger.debug(result)













########################################################################################################
########################################################################################################
# Evaluation of decision calibration: 
# for both precise as well as imprecise models
#############
logger.debug("")
logger.debug("#############")
logger.debug("Beginning evaluation of decision calibration of predictors")
logger.debug("#############")
logger.debug("")


ip_dec_cal_df = []
for predictor in predictors_to_evaluate:
    logger.debug(predictor)

    # evaluation of decision calibration for different loss functions, both on training and on test set
    loss_funcs = cost_sensitive_losses # same as above

    y_preds_train = predictor.predict(X_train_combined)
    y_preds_test = predictor.predict(X_test_combined)


    for l in loss_funcs:
        logger.debug("evaluating decision calibration for %s" % l.name)
        # compute IP decision calibration with respect to the IP data model (each state a distribution)
        
        
        # note: this can be imprecise!
        # if this is imprecise, this will be a numpy array of shape (k,N), where k is the number of ensemble members
        # the number of ensemble members needs to be constant over all predictions
        dec_cal_train_results = [compute_ip_decision_calibration(predictor, data_dictionary[s]["X_train"], \
                                                            data_dictionary[s]["y_train"], l, all_actions=[0.,1.], verbose=False) for s in states]
        
        # this is now a list of 2-tuples: the first term is the unconditonal (without groups) decision calibration term
        # the second is a list which contains the decision calibration terms for each recommended action (dec cal with groups)
        # i.e. with the action-induced partition

        # extract the unconditional term
        dec_cal_train = max([x[0] for x in dec_cal_train_results])

        # for action a=0
        dec_cal_train_a0 = max([x[1][0] for x in dec_cal_train_results])

        # for action a=1
        dec_cal_train_a1 = max([x[1][1] for x in dec_cal_train_results])


        # now for test
        dec_cal_test_results = [compute_ip_decision_calibration(predictor, data_dictionary[s]["X_test"], \
                                                            data_dictionary[s]["y_test"], l, all_actions=[0.,1.], verbose=False) for s in states]
        dec_cal_test = max([x[0] for x in dec_cal_test_results])

        # for action a=0
        dec_cal_test_a0 = max([x[1][0] for x in dec_cal_test_results])

        # for action a=1
        dec_cal_test_a1 = max([x[1][1] for x in dec_cal_test_results])


        ip_dec_cal_df.append((predictor.name, l.name, dec_cal_train, dec_cal_train_a0, dec_cal_train_a1, \
                            dec_cal_test, dec_cal_test_a0, dec_cal_test_a1))




# dec cal with respect to IP (state-wise) data model
ip_dec_cal_df = pd.DataFrame(ip_dec_cal_df, columns=['Predictor Name', 'Loss Name', 'IP DecCal (Train)',\
                    'DecCal (Train) : a=0', 'DecCal (Train) : a=1', 'IP DecCal (Test)', 'DecCal (Test) : a=0', 'DecCal (Test) : a=1'])
logger.debug(ip_dec_cal_df)





########################################################################################################
########################################################################################################
# Save results to dataframes
#############
ipscores_df.to_csv(args.plotdir + 'ipscores_df.csv', index = False)
all_combined_res_df.to_csv(args.plotdir + 'all_combined_res_df.csv', index = False)
ip_dec_cal_df.to_csv(args.plotdir + 'ip_dec_cal_df.csv', index = False)
logger.debug("saved data frames with evaluation results. Call create_plots.py to then create the plots based on these.")
