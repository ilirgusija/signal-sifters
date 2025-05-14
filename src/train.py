import torch
from util.dataset import DichasusDataset
from models.NN import NN
from util.loss import ai_slop
from torch.utils.data import DataLoader
# from util.misc import get_device

def train():
    
    # Instantiate model
    model = NN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  
    loss_fn = ai_slop
    # Instantiate device
    device = 'cpu'
    
    # Instantiate datasets
    labeled_dataset = DichasusDataset("training_data_labeled_np.npz", device=device)
    unlabeled_dataset = DichasusDataset("training_data_unlabeled_np.npz", device=device)
    
    # Example: print dataset sizes
    print(f"Labeled dataset size: {len(labeled_dataset)}")
    print(f"Unlabeled dataset size: {len(unlabeled_dataset)}")
    
    dataloader = DataLoader(labeled_dataset, batch_size=128, shuffle=True)  
    
    model = model.to(device)
    
    # Training loop
    for epoch in range(30):
        print(f"Epoch {epoch} of 30")
        for batch, (csi, pos, _) in enumerate(dataloader):
            # print(f"Batch {batch} of {len(dataloader)}")
            outputs = model(csi)
            loss = loss_fn(outputs, pos[:, :2])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Loss: {loss.item()}")

if __name__ == "__main__":
    train()
