import numpy as np 
import torch
from abc import ABC, abstractmethod



"""
Evaluate decision calibration
# a loss function is of the form ell(a,omega)
# so this is not yet an IP scoring rule
# but if the action space is the space of probabilities, it can be a precise scoring rule
# It induces an entropy as H(P) = E_P[ell(a^*,omega)], where a^* is any bayes optimal action

For numpy use, not by torch
"""
class LossFunction(ABC):
    """
    For predictions of shape (N,), return a Bayes optimal action
    """
    @abstractmethod
    def getBayesAction(self, predictions):
        pass

    """
    Once we know the actions of shape (N,), we can evaluate the loss function
    at outcomes of shape Nx1
    """
    @abstractmethod
    def ell(self, actions, outcomes):
        pass


    """
    For a one-dimensional array of probabilities probs of shape (N,), compute all the individual entropies
    this entails finding the optimal action
    """
    @abstractmethod
    def entropy(self, probs):
        pass


    @abstractmethod
    def getMinMaxAction(self, p0, p1):
        """
        get a minmax optimal action with respect to the imprecise probability interval [p0,p1]
        So this is the action recommendation for a single datum!
        """





class CostSensitiveLoss(LossFunction):

    def __init__(self, c):
        # the only parameter of the cost-sensitive loss function
        self.c = c
        self.name = 'CostSensitive(c=%s)'%c


    def getBayesAction(self, predictions):
        predictions = np.array(predictions)
        assert predictions.ndim == 1
        res = np.ones(len(predictions)) # default action is a=1
        res[predictions<=self.c] = 0. # but sometimes it is a=0, when pred<=c
        return res


    def ell(self, actions, outcomes):
        actions = np.array(actions)
        outcomes = np.array(outcomes)
        assert actions.ndim == 1
        assert outcomes.ndim == 1

        return outcomes*(1-actions)*(1-self.c) + (1-outcomes)*actions*self.c


    def entropy(self, probs):
        probs = np.array(probs)
        assert probs.ndim == 1

        return np.minimum((1-self.c)*probs, self.c*(1-probs))



    def getMinMaxAction(self, p0, p1):
        assert p0<=p1

        # compute max expected loss for each action a=0 and a=1
        # and then take the better one
        # note: maximum entropy may not help here!

        x = self.ell([0.], [1.])
        y = self.ell([1.], [0.])
        ip_score_a0 = max(p0*x, p1*x)
        ip_score_a1 = max((1-p1)*y, (1-p0)*y)


        # tie breaking in favor for recommending a=0
        if ip_score_a0 <= ip_score_a1:
            return 0. # a=0
        else:
            return 1. # a=1


class CostSensitiveLossWithRejectOption(LossFunction):

    def __init__(self, c, r):
        self.c = c # corresponds to parameter of cost weighted loss
        self.r = r # cost of rejecting (abstaining)
        self.name = 'CostSensitiveWithR(c=%s,r=%s)'%(c,r)


    def getBayesAction(self, predictions):
        predictions = np.array(predictions)
        entropies = self.entropy(predictions)
        assert predictions.ndim == 1
        res = np.ones(len(predictions)) # default action is a=1
        res[predictions<=self.c] = 0. # but sometimes it is a=0, when pred<=c

        # when should we abstain? when the entropy is at least r
        res[entropies>=self.r] = 2. # the action 2. is code for abstain/reject
        return res


    def ell(self, actions, outcomes):
        actions = np.array(actions)
        outcomes = np.array(outcomes)
        assert actions.ndim == 1
        assert outcomes.ndim == 1

        # this term works when the action is either 0 or 1
        res = outcomes*(1-actions)*(1-self.c) + (1-outcomes)*actions*self.c
        # but sometimes we reject!
        res[actions==2.] = self.r 
        return res


    def entropy(self, probs):
        probs = np.array(probs)
        assert probs.ndim == 1

        return np.minimum(self.r*np.ones(len(probs)), np.minimum((1-self.c)*probs, self.c*(1-probs)))



    def getMinMaxAction(self, p0, p1):
        assert p0<=p1

        # compute max expected loss for each action a=0 and a=1
        # and then take the better one
        # note: maximum entropy may not help here!

        x = self.ell([0.], [1.])
        y = self.ell([1.], [0.])
        ip_score_a0 = max(p0*x, p1*x)
        ip_score_a1 = max((1-p1)*y, (1-p0)*y)


        # tie breaking in favor for recommending a=0
        if self.r <= min(ip_score_a0,ip_score_a1):
            return 2. # reject
        else:
            # don't reject
            if ip_score_a0 <= ip_score_a1:
                return 0. # a=0
            else:
                return 1. # a=1




class BrierLoss(LossFunction):

    def __init__(self):
        self.name = 'Brier'


    """
    For BrierScore, the optimal action is just the identity function
    """
    def getBayesAction(self, predictions):
        predictions = np.array(predictions)
        assert predictions.ndim == 1
        return predictions


    """
    Simply the squared loss
    """
    def ell(self, actions, outcomes):
        actions = np.array(actions)
        outcomes = np.array(outcomes)
        assert actions.ndim == 1
        assert outcomes.ndim == 1

        return (actions-outcomes)**2


    def entropy(self, probs):
        probs = np.array(probs)
        assert probs.ndim == 1

        return probs*(1-probs)**2 + (1-probs)*probs**2



    def getMinMaxAction(self, p0, p1):
        assert p0<=p1

        if p0 <= 0.5 and p1 >= 0.5:
            return 0.5

        if p1 < 0.5:
            return p1

        if p0 > 0.5:
            return p0





class LogLoss(LossFunction):

    def __init__(self):
        self.name = 'LogLoss'


    """
    For log loss, the optimal action is just the identity function
    """
    def getBayesAction(self, predictions):
        predictions = np.array(predictions)
        assert predictions.ndim == 1
        return predictions


    """
    The log loss
    """
    def ell(self, actions, outcomes):
        actions = np.array(actions)
        outcomes = np.array(outcomes)
        assert actions.ndim == 1
        assert outcomes.ndim == 1


        # for numerical reasons:
        epsilon = 1e-15
        actions = np.clip(actions, epsilon, 1 - epsilon)

        return -outcomes*np.log(actions) + -(1-outcomes)*np.log(1-actions)


    def entropy(self, probs):
        probs = np.array(probs)
        assert probs.ndim == 1


        return probs*self.ell(probs,np.ones(len(probs))) + (1-probs)*self.ell(probs, np.zeros(len(probs)))


    def getMinMaxAction(self, p0, p1):
        assert p0<=p1

        if p0 <= 0.5 and p1 >= 0.5:
            return 0.5

        if p1 < 0.5:
            return p1

        if p0 > 0.5:
            return p0




# from Evaluating Probabilities: Asymmetric Scoring Rules
# winkler, 1994
# note that AsymmetricLoss is not the same as AsymmetricScore here (technically; conceptually the same)
# see below
class AsymmetricLoss(LossFunction):

    def __init__(self, c, smooth=True, smoothing_factor=1000.):
        self.c = c
        self.name = 'AsymmetricLoss(c=%s)'%c
        self.smooth = smooth
        self.smoothing_factor = smoothing_factor

    """
    For asymmetric loss, the optimal action is just the identity function
    """
    def getBayesAction(self, predictions):
        predictions = np.array(predictions)
        assert predictions.ndim == 1
        return predictions


    """
    The asymmetric loss from winkler
    """
    def ell(self, actions, outcomes):
        actions = np.array(actions)
        outcomes = np.array(outcomes)
        assert actions.ndim == 1
        assert outcomes.ndim == 1

        def sigmoid(z):
            return 1./(1. + np.exp(-z))


        def f(preds,outcomes):
            # log loss as base:
            
            """
            def S(actions,outcomes):
                epsilon = 1e-15
                actions = np.clip(actions, epsilon, 1 - epsilon)
                return -outcomes*np.log(actions) + -(1-outcomes)*np.log(1-actions)
            """

            # brier as base:
            S = lambda pred, true : (pred-true)**2
            r = preds

            if self.smooth:
                def smooth_T(c, r, smoothing_factor=self.smoothing_factor):
                    smooth_step = sigmoid(smoothing_factor * (r - c))  # Sigmoid approximation
                    return smooth_step * (S(1., 1.) - S(c, 1.)) + (1 - smooth_step) * (S(0., 0.) - S(c, 0.))

                T = smooth_T(self.c, r)
            else:    
                def T(c, r):
                    return np.where(r >= self.c, S(1., 1.) - S(self.c, 1.), S(0., 0.) - S(self.c, 0.))
                T = T(self.c,r)


            return 1.-(S(r,outcomes)-S(self.c,outcomes))/T

        return f(actions, outcomes)


    # this def of entropy works for any weakly proper scoring rule
    def entropy(self, probs):
        probs = np.array(probs)
        assert probs.ndim == 1


        return probs*self.ell(probs,np.ones(len(probs))) + (1-probs)*self.ell(probs, np.zeros(len(probs)))


    def getMinMaxAction(self, p0, p1):
        raise NotImplementedError







###################################
# MISC



# spherical score in numpy
def SphericalScore(preds,outcomes):
    p_i = preds * outcomes + (1 - preds) * (1 - outcomes)
    print(p_i)

    norm_p = np.sqrt(preds**2 + (1 - preds)**2)
    print(norm_p)

    scores = p_i / norm_p
    return 1.-scores




"""
# asymmetric loss as a simple torch function for training models
# this is the version without smoothing
def AsymmetricScore(c):
    def f(preds,outcomes):
        S = lambda pred, true : (pred-true)**2
        
        #def S(actions,outcomes):
        #    epsilon = torch.tensor(1e-15).double()
        #    actions = torch.clip(actions, epsilon, 1 - epsilon)
        #    return -outcomes*torch.log(actions) + -(1-outcomes)*torch.log(1-actions)
        
        r = preds

        def T(c, r):
            return torch.where(r >= c, S(1., 1.) - S(c, 1.), S(0., 0.) - S(c, 0.))
            #return torch.where(r >= c, S(torch.tensor(1.), torch.tensor(1.)) - S(torch.tensor(c), \
            #        torch.tensor(1.)), S(torch.tensor(0.), torch.tensor(0.)) - S(torch.tensor(c), torch.tensor(0.)))


        return 1.-(S(r,outcomes)-S(torch.tensor(c),outcomes))/T(c,r)
    return f
"""


def AsymmetricScore(c, smoothing_factor=1000.):
    def f(preds, outcomes):
        S = lambda pred, true: (pred - true) ** 2
        r = preds
        
        # Smooth approximation of the step function using a sigmoid
        def smooth_T(c, r, smoothing_factor):
            smooth_step = torch.sigmoid(smoothing_factor * (r - c))  # Sigmoid approximation
            return smooth_step * (S(1., 1.) - S(c, 1.)) + (1 - smooth_step) * (S(0., 0.) - S(c, 0.))
        
        T_val = smooth_T(c, r, smoothing_factor)
        return 1.0 - (S(r, outcomes) - S(torch.tensor(c), outcomes)) / T_val
    
    return f