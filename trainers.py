import numpy as np
import torch
from models import BinaryLogReg_model
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
#import matplotlib.pyplot as plt
import typing
from group_data_loader import GroupDataLoader
from utils import DummyLogger


class Trainer(ABC):
    """
    An abstract class for a trainer  

    Methods
    -------
    train(X_train, y_train)
        train a model on the X_train and y_train data

    transform(X_test)
        Transforms the X_test data so that it can be used as an input for the predict method
        e.g. applying a previously itted standard scaler

    predict(X_test)
        get the precise probabilistic predictions for the X_test data
    """

    @abstractmethod
    def train(self, X_train : np.ndarray, y_train : np.ndarray):
        pass

    @abstractmethod
    def transform(self, X_test : np.ndarray):
        pass

    @abstractmethod
    def predict(self, X_test : np.ndarray):
        pass






class BinaryReg_Trainer(Trainer):
    """
    Very vanilla binary (logistic) regression
    Can use log loss but also other surrogate loss function
    In any case it uses a sigmoid link function due to the underlying BinaryLogReg_model

    Attributes
    -------
    model : torch.nn.Module
        the underlying model to be trained
    n_features : int
        the number of features
    lr : float
        the learning rate
    epochs : int 
        for how many epochs to train
    isTrained : bool
        if the training has been finished
    finalTrainingLoss : flaot
        the final value of the mean surrogate loss
    criterion : Callable
        the surrogate loss function

    Methods
    -------
    train(X_train, y_train, criterion, verbose)
        train a model on the X_train and y_train data, with the surrogate loss function `criterion`

    predict(X_test)
        get the precise probabilistic predictions for the X_test data    
    """
    def __init__(self, n_features : int, lr=1e-3, epochs=3000, logger=DummyLogger()) -> None:
        """
        Initialized the BinaryReg_Trainer
        See the class docstring
        """
        self.model = BinaryLogReg_model(n_features)
        self.n_features = n_features
        self.lr = lr
        self.epochs = epochs
        self.isTrained = False
        self.finalTrainingLoss = None
        self.logger = logger


    def train(self, dataloader : GroupDataLoader, criterion=torch.nn.BCELoss(reduction='none'), verbose=False) -> None:
        """
        Trains the binary (logistic) regressor on the training data

        Parameters:
        -----------
        dataloader : GroupDataLoader
            an instance of the GroupDataLoader class, which we use to obtain the data; in this case, we use a data loader
            which has only a single group (later for the DRO methods, this will be different)
            The dataloader might be one with minibatches or one returning the full dataset at once..but no groups.
        Y_train : numpy.ndarray
            A 1D array of shape (N,1) with the binary class labels
        criterion : Callable
            the surrogate loss function

        Returns:
        --------
        None
        
        Raises:
        -------
        AssertionError
            If the shape number of columns of X_test does not match the prescribed number of features
        """

        self.criterion = criterion
        

        # now the actual training loop
        optimizer=torch.optim.Adam(self.model.parameters(),lr=self.lr)

        # if we wanted to use L-BFGS (full data only!)
        #self.optim = torch.optim.LBFGS(self.model.parameters(), history_size=10, max_iter=4, line_search_fn="strong_wolfe")

        all_losses = []



        loss=None
        for epoch in range(self.epochs):
            
            if epoch % 100 == 0:
                pass
                #dataloader.full_shuffle()

            X_train,y_train = dataloader.getRandomBatches()[0]  # we assume there is only one group here!
            

            
            y_prediction=self.model(X_train)
            y_train = y_train.view(y_train.shape[0],1)

            ind_losses = criterion(y_prediction,y_train).flatten()
            loss = torch.mean(ind_losses)

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            # L-BFGS would work like this:
            """
            def closure(X_train=X_train, y_train=y_train):
                self.optim.zero_grad()
                y_prediction = self.model(X_train)
                y_train = y_train.view(y_train.shape[0],1)

                ind_losses = criterion(y_prediction,y_train).flatten()
                loss = torch.mean(ind_losses)
                if loss.requires_grad:
                    loss.backward()
                return loss
            
            self.optim.step(closure)
            loss = closure()
            if epoch%100 == 0:
                self.logger.debug("loss: %s"%loss.item())
            """


            all_losses.append(float(loss.item()))


            if (epoch)%500 == 0 and verbose:
                self.logger.debug(('epoch:', epoch,',loss=',loss.item()))



        self.logger.debug("training finished.")
        self.isTrained = True
        self.finalTrainingLoss = float(loss.item())



    def predict(self, X_test):
        """
        Predict the precise probabilistic output for the given test data.
        This assumes X_test has already been scaled, if necessary, appropriately

        Parameters:
        -----------
        X_test : numpy.ndarray
            A 2D array of shape (N, number_of_features), where N is the number 
            of samples and number_of_features is the number of features.

        Returns:
        --------
        numpy.ndarray
            The model's probabilistic predictions for the input data, as a numpy array of shape (N,).
        
        Raises:
        -------
        AssertionError
            If the model has not been trained yet (self.isTrained is False).
        """
        assert self.isTrained 
        X_test=torch.from_numpy(X_test).double()
        
        return self.model(X_test).flatten().data.detach().numpy()


    def computeAccuracy(self, X_test, y_test):
        """
        Compute accuracy (with threshold=0.5 for action selection) on test data
        This means effectively using the cost-sensitive loss with c=0.5 as loss function and then computing mean loss

        Parameters:
        -----------
        X_test : numpy.ndarray
            A 2D array of shape (N, number_of_features), where N is the number 
            of samples and number_of_features is the number of features.
        y_test : numpy.ndarray 
            A 1D array of shape (N,), with binary class labels (either 1. or 0.)   

        Returns:
        --------
        float
            The accuracy of the model, when using a threshold at 0.5 (i.e. the cost-sensitive loss with c=0.5)
            on X_test

        Raises:
        -------
        AssertionError
            If the model has not been trained yet (self.isTrained is False).   
        """
        assert self.isTrained 

        y_preds = self.predict(X_test)
        y_preds = torch.from_numpy(y_preds).double()
        y_test=torch.from_numpy(y_test.flatten()).double()

        # we use the action recommended by the cost-sensitive loss with c=0.5 as threshold
        y_pred_class = torch.where(y_preds <= 0.5, torch.tensor(0), torch.tensor(1))
        accuracy=(y_pred_class.eq(y_test).sum())/float(y_test.shape[0])
        return accuracy.item()


    def computeLossDistribution(self, X_test, y_test):
        """
        Compute the loss distribution on test (or train) data
        using the internal surrogate loss, the criterion

        Parameters:
        -----------
        X_test : numpy.ndarray
            A 2D array of shape (N, number_of_features), where N is the number 
            of samples and number_of_features is the number of features.
        y_test : numpy.ndarray 
            A 1D array of shape (N,), with binary class labels (either 1. or 0.)   

        Returns:
        --------
        numpy.ndarray
            A 1D array of shape (N,), containing the loss values of the model's prediction,
             when using the model's criterion (the surrogate loss) to compute the individual losses

        Raises:
        -------
        AssertionError
            If the model has not been trained yet (self.isTrained is False).
        """
        assert self.isTrained


        y_preds = self.predict(X_test)
        y_preds = torch.from_numpy(y_preds).type(torch.float32)
        y_test=torch.from_numpy(y_test.flatten()).type(torch.float32) # long
        ind_losses = self.criterion(y_preds,y_test).flatten()
        return ind_losses.data.detach().numpy().flatten()



    def transform(self, X_test):
        raise NotImplementedError










class BinaryReg_ExpGradDRO_Trainer(BinaryReg_Trainer):
    """
    A DRO trainer for (e.g. logistic, but also other surrogate losses) binary regression
    Uses exponentiated gradient descent to update lambda
    Can use log loss but also other surrogate loss function
    In any case it uses a sigmoid link function due to the underlying BinaryLogReg_model

    Attributes
    -------
    model : torch.nn.Module
        the underlying model to be trained
    n_features : int
        the number of features
    lr : float
        the outer learning rate, concerning the updates of the group weights
    inner_lr : float
        the inner learning rate, concerning the updates of the model's weights
    outer_epochs : int 
        for how many epochs to train, concerning the updates of the group weights
    inner_epochs : int 
        for how many epochs to train, concerning the updates of the model's weights
    isTrained : bool
        if the training has been finished
    criterion : Callable
        the surrogate loss function

    Methods
    -------
    train(X_train, y_train, criterion, verbose)
        train a model on the X_train and y_train data, with the surrogate loss function `criterion`

    predict(X_test)
        get the precise probabilistic predictions for the X_test data    
    """

    def __init__(self, n_features, outer_lr=2., inner_lr=0.1, logger=DummyLogger()):
        self.model = BinaryLogReg_model(n_features)
        self.n_features = n_features
        self.outer_lr = outer_lr
        self.isTrained = False
        self.final_lam = None
        self.inner_lr = inner_lr
        self.logger = logger


    def train(self, dataloader : GroupDataLoader, X_eval: np.ndarray, y_eval : np.ndarray, group_memberships_eval : np.ndarray,\
                 n_groups, criterion=torch.nn.BCELoss(reduction='none'),\
                 verbose=False, outer_epochs=100, inner_epochs = 500, N_batches_eval=10):
        """
        Trains the DRO regressor on the training data
        Following the maximum entropy principle
        In the outer loop (executed outer_epoch many times), the group weights are updated so as to maximize
        In the inner loop, the model parameter's are updated so as to minimize
        So this is max-min, and equals the maximum entropy objective

        _If_ the maximum entropy principle applies, this equals the min-max objective (i.e. the DRO objective)

        Parameters:
        -----------
        dataloader : GroupDataLoader
            use this to get a mini-batch for each group
        X_eval : np.ndarray
            "Test data" on which we test/evaluate the model during training every couple epochs, to see what the DRO loss currently is.
            This can be equal to the training data, but we get the training data from the dataloader!
        y_eval : np.ndarray
            The corresponding test labels. A 1D array of shape (N,) with the binary labels
        group_memberships_eval : np.ndarray
            The corresponding group labels. A 1D array of shape (N,1) with the group labels as floats, where the groups are
            G={0.,1,.,..k}; should always start at zero!
        n_groups : int
            The number k of groups, see below for semantics
            This is important since sometimes we might not have all groups represented in some subset of the data
        criterion : Callale
            the surrogate loss function
        verbose : bool
            whether or not to print more information when training
        outer_epochs : int
            how often to update the group weights
        inner_updates : int
            for each outer epoch, how often to update the model's parameters
        N_batches_eval : int
            how many random batches to use for estimate the gradient for lambda, after an inner loop has been executed

        Returns:
        --------
        None
        
        Raises:
        -------
        AssertionError
            If the shape number of columns of X_test does not match the prescribed number of features
        """

        self.criterion = criterion

        X_eval = torch.from_numpy(X_eval).double()
        y_eval = torch.from_numpy(y_eval).double()
        group_memberships_eval = torch.from_numpy(group_memberships_eval).double()
        

        # get the set of groups
        group_labels = list(range(n_groups))



        # lam is the convex mixture parameter, expressing the weighting of the groups
        self.lam = np.array((np.ones(n_groups)/n_groups).astype(np.float64))
        self.logger.debug("initial lam: %s" % self.lam)


        self.outer_epochs = outer_epochs
        self.inner_epochs = inner_epochs
        self.logger.debug("learning rate: %s" % self.outer_lr)

        # the inner objective
        def minObj(lam):
            lam_param = torch.nn.Parameter(torch.tensor(lam).double(), requires_grad=True)

            min_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.inner_lr)

            # if we want to use L-BFGS:
            #self.optim = torch.optim.LBFGS(self.model.parameters(), history_size=10, max_iter=4, line_search_fn="strong_wolfe")

            dataloader.full_shuffle()

            # in this loop, lam_param is kept fixed
            for epoch in range(self.inner_epochs):
                batches = dataloader.getRandomBatches()

                
                loss = 0.
                for g in group_labels:
                    g = int(g)

                    X_train_g = batches[g][0]
                    y_train_g = batches[g][1]

                    y_prediction_g=self.model(X_train_g)
                    y_train_g = y_train_g.view(y_train_g.shape[0],1)

                    ind_losses = criterion(y_prediction_g,y_train_g).flatten()

                    # now weight the loss by the weight of the group!
                    unscaled_loss = torch.mean(ind_losses)
                    loss += lam_param[g] * unscaled_loss


                lam_param.grad = torch.zeros(n_groups).double()
                min_optimizer.zero_grad()
                loss.backward()
                min_optimizer.step()
                

                # if we want to use L-BFGS:
                # note that L-BFGS should use full data, no batches
                """
                def closure(batches=batches):
                    self.optim.zero_grad()
                    loss = 0.
                    for g in group_labels:
                        g = int(g)

                        X_train_g = batches[g][0]
                        y_train_g = batches[g][1]

                        y_prediction_g=self.model(X_train_g)
                        y_train_g = y_train_g.view(y_train_g.shape[0],1)

                        ind_losses = criterion(y_prediction_g,y_train_g).flatten()

                        # now weight the loss by the weight of the group!
                        unscaled_loss = torch.mean(ind_losses)
                        loss += lam_param[g] * unscaled_loss

                    if loss.requires_grad:
                        loss.backward()
                    return loss

                self.optim.step(closure)
                loss = closure()
                if epoch%100 == 0:
                    self.logger.debug("loss: %s"%loss.item())
                """


            # now we would like to have a good estimate for the subgroup losses
            # for this, using a small mini-batch for each group is not enough - highly variables estimates
            # which are then highly variable gradients for lambda
            # note: the gradient for lambda is simply the subgroup losses!

            with torch.no_grad():
                bigbatches = dataloader.getRandomBigBatches(N_batches_eval)
                loss = 0.
                subgroup_losses = []
                for g in group_labels:
                    g = int(g)

                    X_train_g = bigbatches[g][0]
                    y_train_g = bigbatches[g][1]

                    y_prediction_g=self.model(X_train_g)
                    y_train_g = y_train_g.view(y_train_g.shape[0],1)

                    ind_losses = criterion(y_prediction_g,y_train_g).flatten()
                    subgroup_loss = torch.mean(ind_losses)
                    loss += lam_param[g] * subgroup_loss

                    subgroup_losses.append(float(subgroup_loss.data.detach().numpy()))


            #self.logger.debug("subgroup_losses:")
            #self.logger.debug(subgroup_losses)
            # when inner minimization is done, return the final training loss and the gradient for lambda
            # here we could, if necessary, compute the loss on a much larger batch for each group
            #return loss.item(), lam_param.grad.data.detach().numpy()
            return loss.item(), np.array(subgroup_losses)







        if verbose:
            self.logger.debug("outer lr: %s" % self.outer_lr)

        all_lams = []
        dro_losses = []
        all_losses = []
        for outer_epoch in range(outer_epochs):
            all_lams.append(self.lam)


            if verbose:
                self.logger.debug("");self.logger.debug("");
                self.logger.debug("outer epoch=%s" % outer_epoch)
                self.logger.debug("lam=%s" % self.lam)
            loss, grad = minObj(self.lam)

            if verbose:
                self.logger.debug("loss, grad:")
                self.logger.debug((loss,grad))
            gradrev = -grad

            all_losses.append(loss)


            if outer_epoch % 10 == 0:
                self.logger.debug("weights: %s" % self.lam)
                self.logger.debug("dro loss:")
                group_losses = []
                for g in group_labels:
                    X_train_g = X_eval[group_memberships_eval==g]
                    y_train_g = y_eval[group_memberships_eval==g]
                    y_prediction=self.model(X_train_g)
                    y_train_g = y_train_g.view(y_train_g.shape[0],1)
                    group_loss = torch.mean(criterion(y_prediction,y_train_g).flatten())
                    group_losses.append(float(group_loss.data.detach().numpy()))
                dro_loss = max(group_losses)
                dro_losses.append(dro_loss)
                self.logger.debug(dro_loss)
                self.logger.debug("max-min range: %.6f" % (max(group_losses)-min(group_losses)))


            # now update lam using exponentiated gradient descent
            self.lam = self.lam * np.exp(- self.outer_lr * gradrev) # enforces positivity
            self.lam /= np.sum(self.lam) # enforces unit L1 norm constraint
            # the resulting self.lam is guaranteed to lie again in the probability simplex


            if verbose:
                self.logger.debug("lam after update:")
                self.logger.debug(self.lam)


        if verbose:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.title("DRO losses over outer iterations")
            plt.plot(dro_losses)
            plt.show()

            plt.figure()
            plt.title("all losses")
            plt.plot(all_losses)
            plt.show()

            for i in range(n_groups)[:2]:
                plt.figure()
                plt.title("lam_%s" % i)
                plt.plot([l[i] for l in all_lams])
                plt.show()
        
        self.logger.debug("all DRO losses over time:")
        self.logger.debug(dro_losses)

        if verbose:
            self.logger.debug("training finished.")

        self.final_lam = self.lam
        self.isTrained = True


    def transform(self, X_test):
        raise NotImplementedError





class BinaryReg_MaxEnt_Trainer(BinaryReg_Trainer):
    """
    This trainer computes the maximum entropy distribution explicitly (wrt full Bayes risk)
    it takes two datasets as input, corresponding to joint probabilities P_0 and P_1
    and as lambda varies over [0,1]
    it trains a model for lambda*P_0 + (1-lambda)*P_1

    Attributes
    -------
    model : torch.nn.Module
        the underlying model to be trained
    n_features : int
        the number of features
    epochs : int 
        for how many epochs to train
    isTrained : bool
        if the training has been finished
    criterion : Callable
        the surrogate loss function

    """
    def __init__(self, n_features : int, epochs=3000, logger=DummyLogger()):
        self.model = BinaryLogReg_model(n_features)
        self.n_features = n_features
        self.epochs = epochs
        self.isTrained = False
        self.logger = logger



    #@override
    def train(self, lambdas : np.ndarray, P0_X_train : np.ndarray, P0_y_train : np.ndarray, \
             P1_X_train : np.ndarray, P1_y_train : np.ndarray,\
              criterion=torch.nn.BCELoss(reduction='none')):
        """
        Using data for two groups (two probabilities),
        compute the maximum entropy distribution explicitly
        by considering some lambda's and computing the entropy curve as a one dim concave function

        Parameters:
        -----------
        lambdas : np.ndarray
            the lambdas used to discretize the unit interval as test points
        P0_X_train : numpy.ndarray
            A 2D array of shape (N, number_of_features), where N is the number 
            of samples from the probability P0 and number_of_features is the number of features.
        P0_Y_train : numpy.ndarray
            A 1D array of shape (N,1) with the binary class labels for the probability P0
        P1_X_train : numpy.ndarray
            A 2D array of shape (N, number_of_features), where N is the number 
            of samples from the probability P1nd number_of_features is the number of features.
        P1_Y_train : numpy.ndarray
            A 1D array of shape (N,1) with the binary class labels for the probability P1
        criterion: the surrogate loss function

        Returns:
        --------
        bayes_risks: list[float]
            the corresponding bayes risk results (i.e. the entropies at the lambdas)
        optimal_params : list[tuple(np.ndarray,np.ndarray)]
            the optimal model parameters for each of the lambda's, containing the linear coefficients and the bias term
        
        Raises:
        -------
        AssertionError
            If the shape number of columns of X_test does not match the prescribed number of features
        """
        assert P0_X_train.shape[1] == self.n_features
        assert P1_X_train.shape[1] == self.n_features

        self.criterion = criterion

        # fit and apply a standard scaler
        self.scaler = StandardScaler()
        # fit the scaler on the combined data
        self.scaler.fit(np.vstack([P0_X_train,P1_X_train]))


        P0_X_train_trans = self.scaler.transform(P0_X_train)
        P1_X_train_trans = self.scaler.transform(P1_X_train)

        P0_X_train=torch.from_numpy(P0_X_train_trans).double()
        P0_y_train=torch.from_numpy(P0_y_train.flatten()).double() 

        P1_X_train=torch.from_numpy(P1_X_train_trans).double()
        P1_y_train=torch.from_numpy(P1_y_train.flatten()).double() 

        P0_y_train = P0_y_train.view(P0_y_train.shape[0],1).double()
        P1_y_train = P1_y_train.view(P1_y_train.shape[0],1).double()
        


        bayes_risks = []

        optimal_params = []

        firstlambda = True
        for lam in lambdas:
            # new model every time
            self.model = BinaryLogReg_model(P0_X_train_trans.shape[1])

            self.logger.debug("");
            self.logger.debug("computing bayes risk for lam=%s" % lam)
            if firstlambda:
                epochsnumber = self.epochs # train for longer for the first time
                firstlambda = False
            else:
                epochsnumber = self.epochs 


            self.optim = torch.optim.LBFGS(self.model.parameters(), history_size=10, max_iter=1, line_search_fn="strong_wolfe",\
                tolerance_grad=1e-14,tolerance_change=1e-14)

            gradnorms = []
            def closure(P0_X_train=P0_X_train,P1_X_train=P1_X_train,P0_y_train=P0_y_train,P1_y_train=P1_y_train):
                    self.optim.zero_grad()
                    P0_y_prediction=self.model(P0_X_train)
                    P1_y_prediction=self.model(P1_X_train)

                    P0_ind_losses = criterion(P0_y_prediction,P0_y_train).flatten()
                    P1_ind_losses = criterion(P1_y_prediction,P1_y_train).flatten()

                    loss = lam*torch.mean(P0_ind_losses) + (1.-lam)*torch.mean(P1_ind_losses)

                    if loss.requires_grad:
                        loss.backward()
                        gradnorms.append(torch.norm(self.model.layer1[0].weight.grad))
                    return loss
                    

            # first: L-BFGS
            lowestloss = 999999
            for epoch in range(epochsnumber):
                loss=closure()
                self.optim.step(closure)

                if loss.item()<lowestloss:
                    lowestloss = loss.item()

                if (epoch+1)%200 == 0:
                    # we need to make predictions for all groups
                    # to compute the training loss one each group
                    self.logger.debug(("epoch: %s" % epoch, "loss: %s" % loss.item()))
            

            """
            import matplotlib.pyplot as plt 
            plt.title("lbfgs gradnorms")
            plt.plot(gradnorms)
            plt.show()
            """

            
            # next: Adam
            optimizer=torch.optim.Adam(self.model.parameters(),lr=0.001)
            improved=False
            gradnorms = []
            for epoch in range(epochsnumber):
                P0_y_prediction=self.model(P0_X_train)
                P1_y_prediction=self.model(P1_X_train)

                P0_ind_losses = criterion(P0_y_prediction,P0_y_train).flatten()
                P1_ind_losses = criterion(P1_y_prediction,P1_y_train).flatten()


                loss = lam*torch.mean(P0_ind_losses) + (1.-lam)*torch.mean(P1_ind_losses)

                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                gradnorms.append(torch.norm(self.model.layer1[0].weight.grad))
                
                if loss.item()<lowestloss:
                    if not improved:
                        self.logger.debug("adam improved!!")
                        improved=True
                    lowestloss = loss.item()

                if (epoch+1)%200 == 0:
                    # we need to make predictions for all groups
                    # to compute the training loss one each group
                    self.logger.debug(("epoch: %s" % epoch, "loss: %s" % loss.item()))

            """
            import matplotlib.pyplot as plt 
            plt.title("sgd gradnorms")
            plt.plot(gradnorms)
            plt.show()
            """

            
            self.logger.debug("lowest loss: %s" % lowestloss)
            # the approximation to the bayes risk is the lowest loss that we found  
            bayes_risks.append(lowestloss)
            optimal_params.append((self.model.layer1[0].weight.detach().numpy().flatten(),\
                self.model.layer1[0].bias.detach().numpy().flatten()))
            


        self.isTrained = True

        return bayes_risks, optimal_params


    def transform(self, X_test):
        raise NotImplementedError






def print_optimizer_params(optimizer):
    """
    A helper function, can be useful for debugging optimizers

    Parameters
    -------
    optimizer : torch.optim.Optimizer


    Returns
    -------
    None
        prints to console
    """
    for i, param_group in enumerate(optimizer.param_groups):
        self.logger.debug(f"Parameter group {i}:")
        for key, value in param_group.items():
            if key == 'params':
                self.logger.debug(f"  {key}: {value}")
        self.logger.debug()