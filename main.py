import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from model import Model
from train import train

def main():
   
    # Initialze model with default values
    model = Model()
    x = torch.linspace(0, 1, 10, requires_grad=True).view(-1, 1) # introduce discretized spatial domain
    EA = lambda x : 1 + 0*x
    p = lambda x : 4 * np.pi**2 *torch.sin(2*np.pi*x)

    n_epochs = 10000

    train(
        n_epochs=n_epochs,
        model=model,
        optimizer=optim.Adam(model.parameters()),
        x=x,
        EA=EA,
        p=p,
        u0=0.,
        u1=0.
    )

    
    u_pred = model(x).detach().numpy()
    xplot_disc = x.detach().numpy()
    
    xplot_cont = np.linspace(0, 1, 100)
    u_analytic = np.sin(2*np.pi*xplot_cont)

    plt.plot(xplot_cont, u_analytic, color='b', label='ground-truth')
    plt.scatter(xplot_disc, u_pred, linestyle='dashed', color='r', label='predicted')
    plt.legend()
    plt.title(f"Model predictions after {n_epochs} steps.")
    plt.show()





if __name__ == "__main__":
    main()