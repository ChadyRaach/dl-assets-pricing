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
        loss_fn = loss_fn
        dtype = torch.cuda.FloatTensor

    if not(optimizer) or not(scheduler):
        # Default optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                      betas=(0.9, 0.999), eps=1e-08,
                                      weight_decay=weight_decay, amsgrad=False)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min', factor=0.8, patience=30,
                                                               verbose=True, threshold=0.0001,
                                                               threshold_mode='rel', cooldown=0,
                                                               min_lr=0, eps=1e-08)

    loss_history = []
    valloss_history = []

    # Display initial training and validation loss
    message = ''
    if loader_val is not None:
        valloss = check_accuracy(model, loss_fn, loader_val)
        message = ', val_loss = %.6f' % valloss.item()

    print('Epoch %5d/%5d, ' % (0, num_epochs) + 'loss = %.4f%s' % (-1, message))

    # Save initial results
    # if filename:
    #    torch.save([model, optimizer, loss_history, valloss_history],
    #               filename+'%04d.pt' % 0)

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
            R_pred, _ = model(Z, r, g)
            loss = loss_fn(R, R_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss = epoch_loss / len(loader_train)
        # Store loss history to plot it later
        loss_history.append(epoch_loss)
        if loader_val is not None:
            valloss = check_accuracy(model, loss_fn, loader_val)
            valloss_history.append(valloss)

        if ((epoch + 1) % loss_every == 0):
            message = ''
            if loader_val is not None:
                message = ', val_loss = %.6f' % valloss.item()

            print('Epoch %5d/%5d, ' % (epoch + 1, num_epochs) + 'loss = %.6f%s' % (epoch_loss, message))

        # Save partial results
        if filename and ((epoch + 1) % save_every == 0):
            torch.save([model, optimizer, loss_history, valloss_history],
                       filename + '%04d.pt' % (epoch + 1))
            print('Epoch %5d/%5d, checkpoint saved' % (epoch + 1, num_epochs))

        # scheduler update
        scheduler.step(loss_history[-1])

    # Save last result
    if filename:
        torch.save({"model": model.state_dict(), "opt": optimizer, "train_loss": loss_history,
                   "val_loss": valloss_history}, filename + '%04d.pt' % (epoch + 1))

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
    dtype = torch.FloatTensor
    if torch.cuda.is_available():
        model = model.cuda()
        loss_fn = loss_fn
        dtype = torch.cuda.FloatTensor

    loss = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for (Z, r, g, R) in dataloader:
            Z = Z.type(dtype)
            R = R.type(dtype)
            r = r.type(dtype)
            g = g.type(dtype)
            R_pred, _ = model(Z, r, g)
            loss += loss_fn(R, R_pred)

    return loss / len(dataloader)


if __name__ == "__main__":
    from torch.utils.data.dataloader import DataLoader
    from preprocess_data import next_month
    from preprocess_data import Dataset, train_val_split, batchify
    import pandas as pd
    import numpy as np
    import datetime as dt

    # Load processed file
    firm_carac = pd.read_csv("data/S2000_3300HighestCap.csv")
    ff_portfolio = pd.read_csv("data/ff_portfolios.csv")
    fff = pd.read_csv("data/fama_french_factors.csv")
    filename = ""

    FIRM_CHAR_TO_EXTRACT = ["retx", "me", "be"]
    for df in [firm_carac, ff_portfolio, fff]:
        if type(df["date"].iloc[0]) == str:
            df["date"] = df["date"].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d"))
        df["date"] = df["date"].apply(lambda x: x.replace(day=1))

    start_date = dt.datetime(2000, 1, 1)
    current_month = start_date
    firm_characteristics = []
    firm_returns = []
    benchmark_factors = []
    portfolios_returns = []
    while current_month <= dt.datetime(2021, 12, 31):
        m_fc = firm_carac[(firm_carac["date"] == current_month) & (firm_carac["is_valid"])
                          ].sort_values("me", ascending=False).iloc[:2300]
        assert len(m_fc) == 2300
        firm_characteristics.append(m_fc[FIRM_CHAR_TO_EXTRACT].to_numpy())
        firm_returns.append(m_fc["retx"].to_numpy())
        benchmark_factors.append(fff[fff["date"] == current_month][["mktrf", "smb", "hml"]].to_numpy())
        portfolios_returns.append((ff_portfolio[ff_portfolio["date"] == current_month][[
                                  k for k in ff_portfolio.keys() if "vwret" in k]]).to_numpy())
        current_month = next_month(current_month)
    current_month = current_month - dt.timedelta(days=1)
    firm_characteristics = np.stack(firm_characteristics)
    firm_returns = np.stack(firm_returns)
    benchmark_factors = np.stack(benchmark_factors).squeeze(axis=1)
    portfolios_returns = np.stack(portfolios_returns).squeeze(axis=1)
    ds = Dataset(firm_characteristics[:, :, [1, 2]], firm_returns, benchmark_factors[:, [0, 2]], portfolios_returns,
                 start_date=start_date, end_date=current_month)

    # T = 263
    # M = 2300
    # N = 6
    # r = torch.rand(T, M, 1) * 10
    # g = torch.rand(T, 2, 1)
    # R = torch.rand(T, 5, 1)

    network = SortedFactorModel(3, 2, 4, 1, 2, 6, ranking_method="softmax")
    lambda_ = 0.1

    def loss(gt_returns, pred_returns):
        return pricing_error(gt_returns, pred_returns) + lambda_ * time_series_variation(gt_returns, pred_returns)

    trainsize = int(0.8 * len(ds))
    testsize = len(ds) - trainsize
    train_set, val_set = train_val_split(ds, split_size=0.8)
    trainloader = batchify(train_set, batch_size=12)
    valloader = batchify(val_set, batch_size=12)
    trainmodel(network, loss, trainloader, valloader, None, None, 10, weight_decay=0.1,
               loss_every=2, learning_rate=0.001)
