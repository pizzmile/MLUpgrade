import torch

# Predict labels
def predict_labels(model, unlabeled_loader, device=None):
    model.eval()
    predictions = []
    with torch.no_grad():
        for data in unlabeled_loader:
            if device is not None:
                data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            predictions.extend(pred.cpu().numpy().tolist())
    return predictions