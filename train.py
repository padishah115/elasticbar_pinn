import torch
import torch.optim as optim

def train(n_epochs, model, optimizer:optim.Optimizer, x, EA, p, u0, u1):

    for epoch in range(1, n_epochs+1):
        print(epoch)
        
        f = model.f(x, EA, p)

        x0 = x[0].item()
        x1 = x[-1].item()

        u0_pred = model(torch.tensor([x0]))
        u1_pred = model(torch.tensor([x1]))
        
        MSE_f = torch.sum(f**2) # pde residual loss
        MSE_b = (u0_pred - u0)**2 + (u1_pred-u1)**2 # boundary loss

        loss = MSE_b + MSE_f

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss)