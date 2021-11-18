from tqdm import tqdm
import torch
from model import SortedFactorModel
from losses import pricing_error, time_series_variation


def trainmodel(model, loss_fn, loader_train, loader_val=None,
               optimizer=None, scheduler=None, num_epochs=1,
               learning_rate=0.001, weight_decay=0.0, loss_every=10,
               save_every=10, filename=None):
    """
    function that trains a network model
    Args:
        - model       : network to be trained
        - loss_fn     : loss functions
        - loader_train: dataloader for the training set
        - loader_val  : dataloader for the validation set (default None)
        - optimizer   : the gradient descent method (default None)
        - scheduler   : handles the hyperparameters of the optimizer
        - num_epoch   : number of training epochs
        - learning_rate: learning rate (default 0.001)
        - weight_decay: weight decay regularization (default 0.0)
        - loss_every  : print the loss every n epochs
        - save_every  : save the model every n epochs
        - filename    : base filename for the saved models
    Returns:
        - model          : trained network
        - loss_history   : history of loss values on the training set
        - valloss_history: history of loss values on the validation set
    """

    dtype = torch.FloatTensor
    # GPU
    if torch.cuda.is_available():
        model = model.cuda()
        loss_fn = loss_fn.cuda()
        dtype = torch.cuda.FloatTensor

    if not(optimizer) or not(scheduler):
        # Default optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                      betas=(0.9, 0.999), eps=1e-08,
                                      weight_decay=weight_decay, amsgrad=False)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min', factor=0.8, patience=50,
                                                               verbose=True, threshold=0.0001,
                                                               threshold_mode='rel', cooldown=0,
                                                               min_lr=0, eps=1e-08)

    loss_history = []
    valloss_history = []

    # Display initial training and validation loss
    message = ''
    if loader_val is not None:
        valloss = check_accuracy(model, loss_fn, loader_val)
        message = ', val_loss = %.4f' % valloss.item()

    print('Epoch %5d/%5d, ' % (0, num_epochs) +
          'loss = %.4f%s' % (-1, message))

    # Save initial results
    if filename:
        torch.save([model, optimizer, loss_history, valloss_history],
                   filename+'%04d.pt' % 0)

    # Main training loop
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        # The data loader iterates once over the whole data set
        for (Z, r, g, R) in loader_train:
            # make sure that the models is in train mode
            model.train()

            # Apply forward model and compute loss on the batch
            Z = Z.type(dtype)
            R = R.type(dtype)
            r = r.type(dtype)
            g = g.type(dtype)
            R_pred = model(Z, r, g)
            loss = loss_fn(R_pred, R)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Store loss history to plot it later
        loss_history.append(loss/len(loader_train))
        if loader_val is not None:
            valloss = check_accuracy(model, loss_fn, loader_val)
            valloss_history.append(valloss)

        if ((epoch + 1) % loss_every == 0):
            message = ''
            if loader_val is not None:
                message = ', val_loss = %.4f' % valloss.item()

            print('Epoch %5d/%5d, ' % (epoch + 1, num_epochs) +
                  'loss = %.4f%s' % (loss.item(), message))

        # Save partial results
        if filename and ((epoch + 1) % save_every == 0):
            torch.save([model, optimizer, loss_history, valloss_history],
                       filename+'%04d.pt' % (epoch + 1))
            print('Epoch %5d/%5d, checkpoint saved' % (epoch + 1, num_epochs))

        # scheduler update
        scheduler.step(loss_history[-1].data)

    # Save last result
    if filename:
        torch.save([model, optimizer, loss_history, valloss_history],
                   filename+'%04d.pt' % (epoch + 1))

    return model, loss_history, valloss_history


def check_accuracy(model, loss_fn, dataloader):
    """
    Auxiliary function that computes mean of the loss_fn
    over the dataset given by dataloader.

    Args:
        - model: a network
        - loss_fn: loss function
        - dataloader: the validation data loader

    Returns:
        - loss over the validation set
    """
    import torch

    dtype = torch.FloatTensor
    if torch.cuda.is_available():
        model = model.cuda()
        loss_fn = loss_fn.cuda()
        dtype = torch.cuda.FloatTensor

    loss = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for (Z, r, g, R) in dataloader:
            Z = Z.type(dtype)
            R = R.type(dtype)
            r = r.type(dtype)
            g = g.type(dtype)
            R_pred = model(Z, r, g)
            loss = loss_fn(R_pred, R)

    # return loss divided by number of mini-batches
    return loss/len(dataloader)


if __name__ == "__main__":
    T = 100
    M = 3000
    fake_data = torch.rand(T, M, 52)
    r = torch.rand(T, M, 1) * 10
    g = torch.rand(T, 2, 1)
    R = torch.rand(T, 5, 1)
    network = SortedFactorModel(5, 52, 32, 10, 2, 5)
    lambda_ = torch.Tensor([0.1])

    def loss(gt_returns, pred_returns):
        return pricing_error(gt_returns, pred_returns) + lambda_ * time_series_variation(gt_returns, pred_returns)

    dataloader = [[fake_data, r, g, R]]
    trainmodel(network, loss,
               dataloader, dataloader, None, None, 100, weight_decay=0.1, loss_every=10)
