import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
from skimage.filters import threshold_local
import numpy as np
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingLR

import hypernetx as hnx
import metrics
import pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ImprovedDiffusionModel(nn.Module):
    def __init__(self):
        super(ImprovedDiffusionModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),  # Changed input channels to 2
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x, num_nodes):
        # Expand num_nodes to match the spatial dimensions of x
        num_nodes_channel = num_nodes.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.size(2), x.size(3))
        x_with_nodes = torch.cat([x, num_nodes_channel], dim=1)
        encoded = self.encoder(x_with_nodes)
        decoded = self.decoder(encoded)
        return decoded

def denoise(model, x_noisy, num_nodes):
    return model(x_noisy, num_nodes)

class ImageDataset(Dataset):
    def __init__(self, folder_path, dataset_name, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

        with open('../../datasets/' + dataset_name + '.pkl', 'rb') as file:
            dataset = pickle.load(file)
        self.n_nodes = [len(H.nodes) for H in dataset['train'] for _ in range(5)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_name).convert('L')
        if self.transform:
            image = self.transform(image)
        
        # Get the number of nodes for this image
        num_nodes = self.n_nodes[idx]
        num_nodes = torch.tensor([num_nodes], dtype=torch.float32)
        
        return image, num_nodes

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load the dataset
def load_dataset(folder_name, dataset_name):
    dataset = ImageDataset(folder_path=folder_name, dataset_name=dataset_name, transform=transform)
    return DataLoader(dataset, batch_size=8, shuffle=True)


def train_diffusion_model(folder_name, dataset_name, num_epochs=10, initial_lr=0.001, patience=5, model_save_path='diffusion_model.pth'):
    dataloader = load_dataset(folder_name, dataset_name)
    model = ImprovedDiffusionModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)

    best_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, num_nodes in dataloader:
            images = images.to(device)
            num_nodes = num_nodes.to(device)
            noise = torch.randn_like(images).to(device)
            noisy_images = images + noise
            outputs = denoise(model, noisy_images, num_nodes)
            loss = criterion(outputs, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        scheduler.step()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')

        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    print(f"Best model saved to {model_save_path}")

def adaptive_threshold(tensor, block_size=35, offset=10):
    np_image = tensor.squeeze().cpu().numpy()
    np_image = (np_image - np_image.min()) / (np_image.max() - np_image.min()) * 255
    np_image = np_image.astype(np.uint8)
    #thresh = threshold_local(np_image, block_size, offset=offset)
    binary = np_image > 177
    binary_tensor = torch.from_numpy(binary.astype(np.float32)).unsqueeze(0)
    return binary_tensor

def sample_from_model(model, desired_num_nodes, n_rows, n_cols):
    model.eval()
    samples = []

    for i in range(len(desired_num_nodes)):
        with torch.no_grad():
            sample = torch.randn((1, 1, n_rows, n_cols), device=device)  # Assuming 224x224 images
            num_nodes = torch.tensor([[desired_num_nodes[i]]], dtype=torch.float32).to(device)
            for t in range(256, 0, -1):
                sample = denoise(model, sample, num_nodes)
                sample += torch.randn_like(sample) * 0.1
            samples.append(adaptive_threshold(sample.squeeze().cpu()))

    return samples

def save_samples(samples, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    for i, sample in enumerate(samples):
        save_image(sample, os.path.join(folder_path, f'sample_{i}.png'))

def train_all_datasets(base_data_folder, base_model_folder, num_epochs=10, initial_lr=0.001, patience=5):
    for dataset_folder in Path(base_data_folder).iterdir():
        if dataset_folder.is_dir():
            dataset_name = dataset_folder.name
            train_folder = dataset_folder / 'train'
            
            if train_folder.exists():
                model_save_path = os.path.join(base_model_folder, f'{dataset_name}.pth')
                
                if os.path.exists(model_save_path):
                    print(f"Model for dataset {dataset_name} already exists. Skipping training.")
                    continue
                
                print(f"Training model for dataset: {dataset_name}")
                train_diffusion_model(
                    folder_name=str(train_folder),
                    dataset_name=dataset_name,
                    num_epochs=num_epochs,
                    initial_lr=initial_lr,
                    patience=patience,
                    model_save_path=model_save_path
                )
                print(f"Model for {dataset_name} saved to {model_save_path}")
            else:
                print(f"Warning: Train folder not found for {dataset_name}")

def load_and_generate_samples(base_model_folder, base_data_folder):
    for model_path in Path(base_model_folder).glob('*.pth'):
        dataset_name = model_path.stem
        
        # Skip models ending with 'vae' or 'dcgan'
        if dataset_name.endswith(('vae', 'dcgan')):
            continue
        
        train_folder = Path(base_data_folder) / dataset_name / 'train'
        
        # Load the dataset
        with open('../../datasets/' + dataset_name + '.pkl', 'rb') as file:
            dataset = pickle.load(file)
        
        if not train_folder.exists():
            print(f"Warning: Train folder not found for {dataset_name}")
            continue
        first_image_path = next(train_folder.glob('*.png'), None)
        if first_image_path is None:
            print(f"Warning: No images found in train folder for {dataset_name}")
            continue
        
        with Image.open(first_image_path) as img:
            width, height = img.size
        
        print(f"Processing dataset: {dataset_name}")
        
        model = ImprovedDiffusionModel().to(device)
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        
        samples = sample_from_model(model, [len(H.nodes) for H in dataset['test']], height, width)
        
        output_folder = Path(base_data_folder) / dataset_name / 'generated_samples'
        os.makedirs(output_folder, exist_ok=True)
        save_samples(samples, str(output_folder))
        
        print(f"Generated samples saved in {output_folder}")


# Usage
base_data_folder = 'data'
base_model_folder = 'models'
train_all_datasets(base_data_folder, base_model_folder, num_epochs=1000, initial_lr=0.01, patience=5)


# Usage
base_model_folder = 'models'
base_data_folder = 'data'
load_and_generate_samples(base_model_folder, base_data_folder)



# Define the base path to your dataset
base_path = "data"

# Define the validation metrics
validation_metrics = [
    metrics.NodeNumDiff(),
    metrics.NodeDegreeDistrWasserstein(),
    metrics.EdgeSizeDistrWasserstein(),
    metrics.Spectral(),
    metrics.Uniqueness(),
    metrics.Novelty(),
    metrics.CentralityCloseness(),
    metrics.CentralityBetweenness(),
    metrics.CentralityHarmonic(),
]

# Function to convert an incidence matrix image to a hypernetx hypergraph
def image_to_hypergraph(image_path, n_nodes):
    # Load image
    image = Image.open(image_path).convert('L')

    # Get original image dimensions
    original_width, original_height = image.size

    # Crop the image to maintain the original width but adjust the height
    cropped_image = image.crop((0, 0, original_width, n_nodes))

    # Convert to binary matrix
    matrix = np.array(cropped_image) // 255  # assuming black is 0 and white is 1
    rows, cols = np.where(matrix == 1)

    # Create hyperedges
    hyperedges = {}
    for i, j in zip(rows, cols):
        if j not in hyperedges:
            hyperedges[j] = []
        hyperedges[j].append(i)

    # Convert to hypernetx hypergraph
    hypergraph = hnx.Hypergraph(hyperedges)
    return hypergraph

for dataset_name in os.listdir(base_path):
    dataset_path = os.path.join(base_path, dataset_name)
    if os.path.isdir(dataset_path):
        # Add the ValidEgo metric if "hypergraphEgo" is in the dataset name
        current_metrics = validation_metrics.copy()
        
        if  "hypergraphEgo" in dataset_name:
            current_metrics.append(metrics.ValidEgo())
        
        if "hypergraphSBM" in dataset_name:
            current_metrics.append(metrics.ValidSBM())
        
        if "hypergraphTree" in dataset_name:
            current_metrics.append(metrics.ValidHypertree())
            
        # Load the dataset
        with open('../../data/' + dataset_name + '.pkl', 'rb') as file:
            dataset = pickle.load(file)

        # Collect all hypergraphs in the current dataset
        all_hypergraphs = []
        for i, test_hypergraph in enumerate(dataset['test']):
            n_nodes = len(test_hypergraph.nodes)
            hypergraph = image_to_hypergraph(dataset_path + '/generated_samples/' + f'sample_{i}.png', n_nodes)
            all_hypergraphs.append(hypergraph)
        
        # Compute and print metrics
        print(f"Metrics for dataset {dataset_name}:")
        for metric in current_metrics:
            result = metric(dataset['test'], all_hypergraphs, dataset['train'])
            print(f"{metric}: {result}")
        print("\n" + "="*50 + "\n")
