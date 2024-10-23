import numpy as np

# for precise forecasts
def compute_decision_calibration(preds, outcomes, lossfunc, verbose=False):
    actions = lossfunc.getBayesAction(preds)

    if verbose:
        print("len(outcomes):%s" % len(actions))
        print("a=0: %s" % (actions==0.).sum())
        print("a=1: %s" % (actions==1.).sum())


    losses = lossfunc.ell(actions,outcomes)
    actual_mean_loss = np.mean(losses)

    # now the entropy term
    # with respect to the empirical data model (the empirical distribution)
    # the entropy term is just the mean of the individual entropies
    # where individual entropy means the entropy of an individual prediction
    
    entropy_term = np.mean(lossfunc.entropy(preds))
    
    if verbose:
        print("Term1: actual mean loss (expected loss under empirical data model:")
        print(actual_mean_loss)
        print("Term 2: mean (exp wrt empirical data model) entropy with respect to forecaster:")
        print(entropy_term)
        print("")

    return actual_mean_loss - entropy_term






# for potentially imprecise forecasts
# computes IP
def compute_ip_decision_calibration(predictor, X_test, y_test, lossfunc, all_actions=[], verbose=False):
    """
    TODO: fix the issue which occurs when some action is never recommended!

    """

    ip_preds = predictor.predict(X_test) # can be precise or imprecise!

    actions = predictor.getRecommendedActions(X_test, lossfunc) # this will work in any case

    losses = lossfunc.ell(actions,y_test)
    actual_mean_loss = np.mean(losses)



    # now the second term is more tricky than with precise forecasts
    # we need to compute the imprecise entropy


    entropy_term = None
    # the terms but conditional on each action recommendation
    entropy_terms = []
    if ip_preds.ndim == 1:
        entropy_term = np.mean(lossfunc.entropy(ip_preds))
        entropy_terms = [np.mean(lossfunc.entropy(ip_preds[actions==a])) if (actions==a).sum()>0. else 0. for a in all_actions]

    else:
        # ip preds has shape (K,N), where K is the number of ensemble members
        ents = []
        ents_actions = [[] for a in all_actions]
        for datum_index in range(ip_preds.shape[1]):
            pred = ip_preds[:,datum_index] # the predictions for a specific datum are a column of ip_preds
            p0 = min(pred)
            p1 = max(pred)

            ella_y1 = lossfunc.ell([actions[datum_index]],[1.]) # if outcome=1
            ella_y0 = lossfunc.ell([actions[datum_index]],[0.]) # if outcome=0
            # now compute the upper expectation wrt the prediction of the partial loss
            # the expected loss of a fixed action is a linear function (linear in the probability)
            # thus the max occurs at either p1 or p0!
            t1 = (p1*ella_y1+(1-p1)*ella_y0)
            t0 = (p0*ella_y1+(1-p0)*ella_y0)
            single_entropy = max(t1,t0)
            ents.append(single_entropy)
            ents_actions[all_actions.index(actions[datum_index])].append(float(single_entropy))

        entropy_term = np.mean(ents)


    if verbose:
        print("Term1: actual mean loss (expected loss under empirical data model:")
        print(actual_mean_loss)
        print("Term 2: mean (exp wrt empirical data model) entropy with respect to forecaster:")
        print(entropy_term)
        print("")




    # now we also do this for the action induced partition with the given actions
    # typically, actions=[0.,1.] are the binary actions
    # first, collect the first terms (the actual mean losses) but conditional on all the action recommendations

    actual_mean_losses = [np.mean(losses[actions==a]) if (actions==a).sum()>0 else 0. for a in all_actions]
    if ip_preds.ndim > 1:
        entropy_terms = [np.mean(ent_list) for ent_list in ents_actions]
    action_dec_cal_terms = [a-b for (a,b) in zip(actual_mean_losses,entropy_terms)]

    

    return actual_mean_loss - entropy_term, action_dec_cal_terms






