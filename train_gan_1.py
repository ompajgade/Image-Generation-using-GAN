import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from generator_1 import Generator
from discriminator_1 import Discriminator

# Training parameters
batch_size = 64
lr = 0.0002
epochs = 500

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Initialize models
G = Generator().cuda()
D = Discriminator().cuda()

# Loss and optimizer
criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr)
optimizer_D = optim.Adam(D.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    for real_imgs, _ in train_loader:
        real_imgs = real_imgs.cuda()
        
        # Train Discriminator
        optimizer_D.zero_grad()
        z = torch.randn(real_imgs.size(0), 100).cuda()  # Generate random noise
        fake_imgs = G(z)
        
        real_labels = torch.ones(real_imgs.size(0), 1).cuda()
        fake_labels = torch.zeros(real_imgs.size(0), 1).cuda()
        
        real_loss = criterion(D(real_imgs), real_labels)
        fake_loss = criterion(D(fake_imgs.detach()), fake_labels)
        d_loss = real_loss + fake_loss
        
        d_loss.backward()
        optimizer_D.step()
        
        # Train Generator
        optimizer_G.zero_grad()
        g_loss = criterion(D(fake_imgs), real_labels)  # Trick discriminator
        g_loss.backward()
        optimizer_G.step()
    
    print(f"Epoch [{epoch+1}/{epochs}]  D Loss: {d_loss.item()}  G Loss: {g_loss.item()}")

# Save the trained generator model
torch.save(G.state_dict(), 'generator.pth')