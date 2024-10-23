import torch
import numpy as np
from abc import ABC, abstractmethod
import math
from enum import Enum



class GroupDataLoader(ABC):
    def __init__(self):
        pass

    def getRandomBatches(self):
        pass 

    def getRandomBigBatches(self, N):
        pass 

    def full_shuffle(self):
        pass




# single groups, always returns full data
class TrivialDataLoader(GroupDataLoader):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).double()
        self.y = torch.from_numpy(y.flatten()).double()
        self.full_shuffle()


    def getRandomBatches(self):
        return [(self.X,self.y)]

    def getRandomBigBatches(self, N):
        return self.getRandomBatches()

    def full_shuffle(self):
        indices = np.random.permutation(self.X.shape[0])
        self.X = self.X[indices]
        self.y = self.y[indices]



# no mini batches, always returns whole data for each group
# we always assume group labels are of the form {0.,..,k.}, if there are k+1 many groups
class FullGroupDataLoader(GroupDataLoader):
    def __init__(self, X : np.ndarray, y : np.ndarray, group_memberships : np.ndarray, n_groups : int):
        self.X = torch.from_numpy(X).double()
        self.y = torch.from_numpy(y.flatten()).double()

        self.group_memberships = group_memberships
        self.group_labels = np.arange(n_groups)

        self.group_indices = {g: np.where(self.group_memberships == g)[0] for g in self.group_labels}



    def full_shuffle(self):
        pass
        """
        # for each group, we fully shuffle its data
        # this could only be useful if we have some training method where the order could matter
        # for mean loss, the order does not matter, so calling this method is unnecessary
        indices = np.random.permutation(self.X.shape[0])

        # Apply the permutation to shuffle X and y and the group labels jointly
        self.X = self.X[indices]
        self.y = self.y[indices]
        self.group_memberships = self.group_memberships[indices]
        self.group_indices = {g: np.where(self.group_memberships == g)[0] for g in self.group_labels}
        """

    def getRandomBatches(self):
        # simply return the full data for each group
        if len(self.group_labels)>1:
            return [(self.X[self.group_indices[g]], self.y[self.group_indices[g]]) for g in self.group_labels]
        else:
            return [(self.X,self.y)]


    def getRandomBigBatches(self, N):
        return self.getRandomBatches()



    """
    def __iter__(self):
        #Reset the iteration and return the iterator object.
        self.current_batch = 0
        self._shuffle_data()
        return self


    def __next__(self):
        # this will only return the full dataset _once_
        # i.e. there is a single batch
        if self.current_batch > 0:
            raise StopIteration

        self.current_batch += 1
        if len(self.group_labels)>1:
            return [(self.X[self.group_indices[g]], self.y[self.group_indices[g]]) for g in self.group_labels]
        else:
            return [(self.X,self.y)]
    """
        



class MiniBatchGroupDataLoader(GroupDataLoader):
    def __init__(self, X : np.ndarray, y : np.ndarray, group_memberships : np.ndarray, n_groups : int, batch_size : int, verbose=False):
        self.X = torch.from_numpy(X).double()
        self.y = torch.from_numpy(y.flatten()).double()

        self.group_memberships = group_memberships
        self.group_labels = np.arange(n_groups)

        self.batch_size = batch_size

        self.group_indices = {g: np.where(group_memberships == g)[0] for g in self.group_labels}

        # for each group, we compute the batch size so that each batch has the full batch_size
        self.batch_numbers = [len(self.group_indices[g]) // batch_size for g in self.group_labels]

        self.verbose = verbose 

        print("training data: ")
        print(X.shape)
        print("batch numbers:")
        print(self.batch_numbers)

        self.full_shuffle()
        self._compute_indices()

    def getRandomBatches(self):
        # for each of the group, get a mini-batch at random
        rnd_indices = [np.random.choice(range(self.batch_numbers[g])) for g in self.group_labels]
        return [(self.X[self.all_batch_indices[g][i]],self.y[self.all_batch_indices[g][i]]) for i,g in zip(rnd_indices, self.group_labels)]


    def getRandomBigBatches(self, N):
        multiple_batches = [self.getRandomBatches() for i in range(N)]
        # now stack them for each group
        N_groups = len(multiple_batches[0]) # could use any index here
        stacked = [(torch.vstack([blist[j][0] for blist in multiple_batches]), torch.hstack([blist[j][1] for blist in multiple_batches])) for j in range(N_groups)]
        return stacked


    def full_shuffle(self):
        """Shuffle the data within each group."""
        for g in self.group_labels:
            indices = self.group_indices[g]
            self.group_indices[g] = indices[torch.randperm(len(indices))]

        #print("group_indices:")
        #print(self.group_indices)
        self._compute_indices()



    def _compute_indices(self):
        # after shuffling, precompute the batch indices for each group

        # a dict where keys are groups and values are dicts with the batch indices
        # say there are k many groups:
        # e.g. all_batch_indices = {0 : {0: indices for group 0 and batch 0, 1: indices for group 0 and batch 1, ...}, .. , k : ..}
        self.all_batch_indices = {}
        for g in self.group_labels:
            batch_indices = {}
            for curr_index in range(self.batch_numbers[g]):
                start_idx = curr_index * self.batch_size
                end_idx = start_idx + self.batch_size
                batch_indices[curr_index] = self.group_indices[g][start_idx:end_idx]
            self.all_batch_indices[g] = batch_indices


