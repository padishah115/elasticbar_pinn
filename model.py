import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

class Model(nn.Module):

    def __init__(self, input_dims=1, hidden_dims=10, output_dims=1):
        """
        Parameters
        ----------
            input_dims : int
                The number of dimensions (nodes) for the first linear (input) layer of the model.
            hidden_dims : int
                The number of dimensions (nodes) in the model's hidden layer.
            output_dims : int
                The number of dimensions (nodes) in the model's final (output) layer.
        """

        super().__init__()

        torch.manual_seed(42)

        # Architecture
        self.fc1 = nn.Linear(in_features=input_dims, out_features=hidden_dims)
        self.activation = F.tanh
        self.fc2 = nn.Linear(in_features=hidden_dims, out_features=output_dims)


    def forward(self, x:torch.tensor)->torch.Tensor:
        """Model forwards pass.
        
        Parameters 
        ----------
            x : torch.Tensor
                Model input in tensor form.
        
        """

        out = self.fc2(self.activation(self.fc1(x)))
        return out
    
    def get_derivative(self, y, x)->tuple[torch.Tensor]:
        """Returns the derivative of y with respect to x for some input y, which is a function of some other input x.
        
        Parameters
        ----------
            y : torch.tensor
                Some function of x, the independent variable.
            x : torch.tensor
                Independent variable.
        """

        dydx = grad(
            outputs = y,
            inputs = x,
            grad_outputs = torch.ones(x.size()[0], 1),
            create_graph=True,
            retain_graph=True
        )[0]

        return dydx

    def f(self, x:torch.Tensor, EA, p):
        """Function for calculating f, as defined to be the differential equation.
        
        Parameters:
        -----------
            x : torch.tensor
                Model input in tensor form.
            EA
                Young modulus multiplied by area, itself a function of x
            p : float
                Inhomogeneous forcing term for the differential equation.
        """

        u = self.forward(x)
        u_x = self.get_derivative(u, x)
        EAu_xx = self.get_derivative(EA(x)*u_x, x)

        f = EAu_xx + p(x)

        return f