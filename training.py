import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Hyperparameters
batch_size = 128
lr = 0.0002
epochs = 200
z_dim = 100
image_size = 28 * 28  # MNIST images are 28x28 pixels
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Generator
class Generator(nn.Module):
    def __init__(self, z_dim, image_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, image_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, image_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(image_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Initialize the Generator and Discriminator
netG = Generator(z_dim, image_size).to(device)
netD = Discriminator(image_size).to(device)

# Optimizers
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCELoss()

# Store losses for plotting
d_losses = []
g_losses = []

# Training loop
for epoch in range(epochs):
    for i, (images, _) in enumerate(train_loader):
        current_batch_size = images.size(0)

        # Prepare real images
        real_images = images.view(current_batch_size, -1).to(device)
        real_labels = torch.ones(current_batch_size, 1).to(device)

        # Train Discriminator with real images
        optimizerD.zero_grad()
        real_outputs = netD(real_images)
        d_loss_real = criterion(real_outputs, real_labels)
        d_loss_real.backward()

        # Generate fake images
        z = torch.randn(current_batch_size, z_dim).to(device)
        fake_images = netG(z)
        fake_outputs = netD(fake_images.detach())
        fake_labels = torch.zeros(current_batch_size, 1).to(device)

        # Train Discriminator with fake images
        d_loss_fake = criterion(fake_outputs, fake_labels)
        d_loss_fake.backward()
        optimizerD.step()

        d_loss = d_loss_real + d_loss_fake

        # Train Generator
        optimizerG.zero_grad()
        gen_outputs = netD(fake_images)
        g_loss = criterion(gen_outputs, real_labels)
        g_loss.backward()
        optimizerG.step()

        # Store losses
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')



# Plotting the comparison of G and D losses
plt.figure(figsize=(10, 5))
plt.title("Comparison of Generator and Discriminator Loss During Training")
plt.plot(d_losses, label="Discriminator")
plt.plot(g_losses, label="Generator")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plotting the discriminator losses
plt.figure(figsize=(10, 5))
plt.title("Discriminator Loss During Training")
plt.plot(d_losses, label="Discriminator Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plotting the generator losses
plt.figure(figsize=(10, 5))
plt.title("Generator Loss During Training")
plt.plot(g_losses, label="Generator Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
# Save the models
torch.save(netG, 'generator.pt')
torch.save(netD.state_dict(), 'discriminator.pt')
torch.save(netG.state_dict(), 'generator_weights.pt')
