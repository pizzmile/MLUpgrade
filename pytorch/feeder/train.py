import torch
import copy

from ...utils import Timer

# Train model
def train_model(model, criterion, optimizer, train_loader, eval_loader, scheduler = None, num_epochs: int = 25, stopper = None, device = None):
    main_timer = Timer()
    main_timer.start()
    epoch_timer = Timer()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        epoch_timer.reset()
        epoch_timer.start()
        
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        stop = False
        for phase in ['train', 'eval']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            if phase == 'train':
                data_loader = train_loader
            else:
                data_loader = eval_loader
                
            for inputs, labels in data_loader:
                if device is not None:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'train' and scheduler is not None:
                scheduler.step()
                
            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.double() / len(data_loader.dataset)
            
            partial = epoch_timer.partial()
            print('{} Loss: {:.4f} | Accuracy: {:.4f} | Duration: {:.4f} [s]'.format(
                phase, epoch_loss, epoch_acc, partial))
            
            if phase == 'eval':
                stop = stopper.step(epoch_acc)
            
            # Deep copy the model
            if phase == 'eval' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
               
        delta = epoch_timer.stop()
        print('Epoch duration: {:.4f} [s]'.format(delta))
        print()
        
        if stop:
            print('Early stop!')
            break;
        
    time_elapsed = main_timer.stop()
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model