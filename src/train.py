import torch
from util.dataset import DichasusDataset
from models.NN import EnsembleNN
from util.loss import ai_slop, old_ai_slop, compute_localization_metrics
from torch.utils.data import DataLoader
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

def train(epochs=30, initial_lr=0.0005, weight_decay=1e-4, alpha=0.5, num_models=8):
    # alpha is the weight of the MAE in the loss function (0.5 means equal weight)

    # Instantiate model
    model = EnsembleNN(num_models=num_models)
    print(f"Number of trainable parameters: {model.count_parameters():,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    # Add learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    loss_fn = ai_slop
    # Instantiate device
    device = 'cpu'
    
    # Instantiate datasets and dataloaders
    labeled_dataset = DichasusDataset("training_data_labeled_np.npz", device=device)
    unlabeled_dataset = DichasusDataset("training_data_unlabeled_np.npz", device=device)
    test_dataset = DichasusDataset("test_data_np.npz", device=device)

    # Example: print dataset sizes
    print(f"Labeled dataset size: {len(labeled_dataset)}")
    print(f"Unlabeled dataset size: {len(unlabeled_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    dataloader = DataLoader(labeled_dataset, batch_size=128, shuffle=True)  
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = model.to(device)
    
    # Compute normalization parameters from training data
    model.set_normalization_params(dataloader)
    
    # Keep track of best test loss
    best_test_loss = float('inf')
    
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch} of {epochs}")
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch, (csi, pos, _) in enumerate(dataloader):
            # Get both ensemble and individual predictions
            ensemble_pred, individual_preds = model.forward_all(csi)
            
            # Calculate loss for ensemble prediction
            ensemble_loss = loss_fn(ensemble_pred, pos[:, :2])
            
            # Calculate loss for each individual model
            individual_losses = torch.stack([loss_fn(pred, pos[:, :2]) for pred in individual_preds])
            
            # Total loss is ensemble loss plus mean of individual losses
            loss = ensemble_loss + torch.mean(individual_losses)
            
            optimizer.zero_grad()
            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
            
        avg_train_loss = epoch_loss / num_batches
        print(f"Training loss: {avg_train_loss:.6f}")
        
        # Test model using batches instead of all at once
        model.eval()
        test_loss = 0.0
        num_test_batches = 0
        with torch.no_grad():
            for test_batch, (test_csi, test_pos, _) in enumerate(test_dataloader):
                # Only use ensemble prediction for validation
                test_predicted_pos = model(test_csi)
                batch_loss = loss_fn(test_predicted_pos, test_pos[:, :2])
                test_loss += batch_loss.item()
                num_test_batches += 1
        
        avg_test_loss = test_loss / num_test_batches
        print(f"Testing loss: {avg_test_loss:.6f}")
        
        # Update learning rate based on test loss
        scheduler.step(avg_test_loss)
        
        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'test_loss': avg_test_loss,
            }, 'best_model.pth')
        
        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.2e}")
        
    # Compute final metrics
    model.eval()
    with torch.no_grad():
        # Get both ensemble and individual predictions for analysis
        final_ensemble_pred, final_individual_preds = model.forward_all(test_dataset.csi)
        
        # Compute metrics for ensemble prediction
        final_ensemble_outputs = final_ensemble_pred.detach().cpu().numpy()
        final_labels = test_dataset.pos[:, :2].detach().cpu().numpy()
        final_ensemble_loss = compute_localization_metrics(final_ensemble_outputs, final_labels)
        
        print("\nEnsemble Model Metrics:")
        print(f"Final MAE: {final_ensemble_loss[0]:.6f}")
        print(f"Final R90: {final_ensemble_loss[1]:.6f}")
        print(f"Score: {1/2*(final_ensemble_loss[0] + final_ensemble_loss[1]):.6f}")
        
        # Compute metrics for individual models
        print("\nIndividual Model Metrics:")
        for i in range(num_models):
            individual_outputs = final_individual_preds[i].detach().cpu().numpy()
            individual_loss = compute_localization_metrics(individual_outputs, final_labels)
            print(f"\nModel {i+1}:")
            print(f"MAE: {individual_loss[0]:.6f}")
            print(f"R90: {individual_loss[1]:.6f}")
            print(f"Score: {1/2*(individual_loss[0] + individual_loss[1]):.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the neural network model')
    parser.add_argument('--epochs', type=int, default=50,
                      help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.0005,
                      help='initial learning rate (default: 0.0005)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                      help='weight decay/L2 regularization (default: 0.0001)')
    parser.add_argument('--alpha', type=float, default=0.5,
                      help='weight of the MAE in the loss function (default: 0.5)')
    parser.add_argument('--num-models', type=int, default=8,
                      help='number of models in the ensemble (default: 8)')
    args = parser.parse_args()
    train(epochs=args.epochs, initial_lr=args.lr, weight_decay=args.weight_decay, 
          alpha=args.alpha, num_models=args.num_models)
