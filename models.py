import torch
import typing
import numpy as np


class BinaryLogReg_model(torch.nn.Module):
    """
    A binary logistic regression model using a single linear layer with ReLU activation 
    and a sigmoid output.

    Parameters
    ----------
    no_input_features : int
        The number of input features for the linear layer.

    Attributes
    ----------
    layer1 : torch.nn.Sequential
        A sequential container with a single linear layer.

    Methods
    -------
    forward(x)
        Forward pass through the model, returning the sigmoid of the linear layer's output.
    """
    def __init__(self,no_input_features : int) -> None:
        """
        Initializes the BinaryLogReg_model with the specified number of input features.

        Parameters
        ----------
        no_input_features : int
            The number of input features for the linear layer.
        """
        super(BinaryLogReg_model,self).__init__()
        out_dim = 1
        
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(no_input_features, out_dim)
        ).double()
        
        
    def forward(self,x : np.ndarray) -> np.ndarray:
        """
        Defines the forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape (N, no_input_features).

        Returns
        -------
        torch.Tensor
            The output tensor after applying the linear layer and sigmoid activation function, 
            with values between 0 and 1.
        """
        y_predicted=self.layer1(x)
        return torch.sigmoid(y_predicted)




class BinaryLogReg_model_withHiddenLayer(torch.nn.Module):
    #   class BinaryLogReg_model(torch.nn.Module):
    """
    A binary logistic regression model, with a single hidden layer
    and a sigmoid output.

    Parameters
    ----------
    no_input_features : int
        The number of input features for the linear layer.

    Attributes
    ----------
    layer1 : torch.nn.Sequential
        A sequential container with a single linear layer.

    layer2 : torch.nn.Sequential
        A sequential container with a single linear layer.

    Methods
    -------
    forward(x)
        Forward pass through the model, returning the sigmoid of the linear layer's output.
    """

    def __init__(self, no_input_features : int) -> None:
        """
        Initializes the BinaryLogReg_model with the specified number of input features.

        Parameters
        ----------
        no_input_features : int
            The number of input features for the linear layer.
        """
        super(BinaryLogReg_model, self).__init__()
        hidden_dim = 8 
        out_dim = 1
        
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(no_input_features, hidden_dim),
            torch.nn.ReLU()
        ).double()
        
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, out_dim)
        ).double()
        

    def forward(self, x : np.ndarray) -> np.ndarray:
        """
        Defines the forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape (N, no_input_features).

        Returns
        -------
        torch.Tensor
            The output tensor after applying the linear layer and sigmoid activation function, 
            with values between 0 and 1.
        """
        x = self.layer1(x)
        y_predicted = self.layer2(x)
        return torch.sigmoid(y_predicted)