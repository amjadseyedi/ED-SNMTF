import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import matplotlib.pyplot as plt
from Utils import SVCA, orthNNLS

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# NeuralMF Model
class NeuralMF(nn.Module):
    def __init__(self, X, latent_dims):
        super(NeuralMF, self).__init__()
        self.n = X.shape[0]
        self.latent_dims = latent_dims
        self.p = len(latent_dims) - 1

        options = {'average': 1}
        Xnp = X.cpu().numpy()

        W0 = SVCA(Xnp, k, k, options=options).astype(np.float32)
        norm2x = np.sqrt(np.sum(Xnp ** 2, axis=0))
        Xn = Xnp * (1 / (norm2x + 1e-16))
        W0 = orthNNLS(Xnp, W0, Xn).T

        self.weights = nn.ParameterList([
            nn.Parameter(torch.tensor(W0.astype(np.float32), device=device))
            for i in range(self.p)
        ])

        self.S = nn.Parameter(torch.eye(latent_dims[-1], device=device))

    def forward(self):
        W_prod = self.S
        for W in reversed(self.weights):
            W_prod = W_prod @ W.T
        for W in reversed(self.weights):
            W_prod = W @ W_prod
        return W_prod

    def loss(self, X, k, lam):
        X_hat = self.forward()
        S_hat = self.weights[0].T @ X @ self.weights[0]

        recon_loss = torch.norm(X - X_hat, p=2)**2
        encoder_loss = lam * torch.norm(self.S - S_hat, p=2) ** 2

        # Final embedding
        W_pos = torch.eye(self.n, device=device)
        for W in self.weights:
            W_pos = W_pos @ W

        total_loss = (recon_loss + encoder_loss)

        return total_loss, recon_loss, encoder_loss

    def apply_nonnegativity_constraint(self):
        with torch.no_grad():
            self.S.data.clamp_(min=0)
            for i in range(len(self.weights)):
                self.weights[i].data.clamp_(min=0)

dataset = 'Email'
data = torch.load(f'{dataset}.pt')

A = data['A'].float().to(device)
y = data['y']
n = A.shape[0]
c= len(np.unique(y))
latent_dims = [n, c]
k = c  # Number of clusters

lam = 0.0001
lr = 0.01

# Model and optimizer
model = NeuralMF(A, latent_dims).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
num_epochs = 1000

err = torch.zeros(num_epochs, device=device)

# Training loop
for epoch in range(num_epochs):

    optimizer.zero_grad()

    loss, recon_loss, encoder_loss = model.loss(A, k, lam)
    loss.backward()
    optimizer.step()
    model.apply_nonnegativity_constraint()
    err[epoch] = torch.sqrt(loss.item()/torch.norm(A)**2)

    pred = torch.argmax(model.weights[0], dim=1).cpu().numpy()
    nmi = normalized_mutual_info_score(y, pred)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Total Loss: {loss.item():.4f}, "
              f"Recon: {recon_loss.item():.4f}, "
              f"Encoding: {encoder_loss.item():.4f}, "
              f'NMI: {nmi:.4f}'
              )

# Evaluation
W_prod = torch.eye(n, device=device)
for W in model.weights:
    W_prod = W_prod @ W

pred = torch.argmax(W_prod, dim=1).cpu().numpy()
nmi = normalized_mutual_info_score(y, pred)
print(f'NMI: {nmi:.4f}')

plt.plot(err.cpu().numpy())
plt.xlabel('Epoch')
plt.ylabel('Loss value')
plt.grid(True)
plt.show()
