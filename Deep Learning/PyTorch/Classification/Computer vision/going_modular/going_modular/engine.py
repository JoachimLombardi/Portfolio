from typing import Dict, List
from timeit import default_timer as timer
import torch
from tqdm.auto import tqdm

def train_test_step(model: torch.nn.Module, 
                    train_dataloader: torch.utils.data.DataLoader, 
                    test_dataloader: torch.utils.data.DataLoader, 
                    loss_fn: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    device: torch.device,
                    epochs: int = 5) -> Dict[str, List[float]]:
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    torch.manual_seed(42)
    train_start = timer()
    for epoch in tqdm(range(epochs)):
        print(f"Epoch {epoch}\n-------")
        train_loss = 0
        train_acc = 0
        for batch, (X,y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            model.train().to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss
            train_acc += ((torch.eq(torch.argmax(dim=1, input=y_pred), y)).sum().item()/len(y_pred))*100
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % 400 == 0:
                print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")
        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)
        test_loss = 0
        test_acc = 0
        model.eval().to(device)
        with torch.inference_mode():
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                test_pred = model(X)
                test_loss += loss_fn(test_pred, y)
                test_acc += ((torch.eq(torch.argmax(dim=1, input=test_pred), y)).sum().item()/len(test_pred))*100
            test_loss /= len(test_dataloader)
            test_acc /= len(test_dataloader)
        results["train_loss"].append(train_loss.item())
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss.item())
        results["test_acc"].append(test_acc)
        print(f"Train loss: {train_loss:.3f} | Train accuracy: {train_acc:.2f}%")
        print(f"Test loss: {test_loss:.3f} | Test accuracy: {test_acc:.2f}%")
    train_end = timer()
    total_time = train_end - train_start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return results
