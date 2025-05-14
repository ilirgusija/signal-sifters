import torch
from util.dataset import DichasusDataset
from models.NN import NN
from util.loss import train_loss
from torch.utils.data import DataLoader
from util.misc import get_device

def train():
    # Instantiate datasets
    labeled_dataset = DichasusDataset(npz_file="training_data_labeled_np.npz")
    unlabeled_dataset = DichasusDataset(npz_file="training_data_unlabeled_np.npz")

    # Example: print dataset sizes
    print(f"Labeled dataset size: {len(labeled_dataset)}")
    print(f"Unlabeled dataset size: {len(unlabeled_dataset)}")
    
    # Instantiate model
    model = NN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  
    loss_fn = train_loss
    dataloader = DataLoader(labeled_dataset, batch_size=128, shuffle=True)  
    
    # Instantiate device
    device = get_device()
    model.to(device)
    dataloader.to(device)
    
    # Training loop
    for epoch in range(30):
        for batch, truth in dataloader:
            outputs = model(batch)
            loss = loss_fn(outputs, truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    train()
