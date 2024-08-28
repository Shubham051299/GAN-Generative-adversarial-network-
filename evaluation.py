import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Assuming the Generator class is defined in a module named 'training'
# If the Generator class is in this script, you can remove the import and define the class directly


# Define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Generator
class Generator(nn.Module):
    def __init__(self, z_dim, image_dim):  # Corrected __init__ method
        super(Generator, self).__init__()  # Corrected super().__init__()
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
def load_model(model_path='./generator_weights.pt', z_dim=100):
    model = Generator(z_dim, 28*28).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def generate_images(model, num_images=100, z_dim=100):
    '''
    Generate a set of images from the model.
    '''
    z = torch.randn(num_images, z_dim, device=device)
    with torch.no_grad():
        images = model(z).cpu().numpy()
    return images.reshape(num_images, 28, 28)  # Reshape to image dimensions if necessary

def save_images(images, filename='generated_images.png'):
    '''
    Save the generated images to a file.
    '''
    fig, axs = plt.subplots(10, 10, figsize=(10,10))  
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(images[i], cmap='gray')
        ax.axis('off')
    plt.savefig(filename)

def plot_images(images):
    '''
    Display the generated images.
    '''
    fig, axs = plt.subplots(10, 10, figsize=(10, 10))  
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(images[i], cmap='gray')
        ax.axis('off')
    plt.show()

if __name__ == "__main__":
    model_path = './generator_weights.pt'  # Path to your generator model
    num_images = 100  # Number of images to generate
    z_dim = 100  # Dimension of the latent space

    model = load_model(model_path, z_dim)
    images = generate_images(model, num_images, z_dim)
    save_images(images)  # Save generated images to a file
    plot_images(images)  # Plot the generated images
