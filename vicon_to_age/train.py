import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb

from learning3d.data_utils import UserData
from learning3d.models import PointNet
from learning3d.models import Classifier
from data_reader import read
import config



def test_one_epoch(device, model, test_loader):
    model.eval()
    test_loss = 0.0
    pred  = 0.0
    count = 0
    for i, data in enumerate(tqdm(test_loader)):
        points, target = data
        target = target[:,0]

        points = points.to(device)
        target = target.to(device)

        output = model(points)
        loss_val = torch.nn.functional.nll_loss(
            torch.nn.functional.log_softmax(output, dim=1), target, size_average=False)

        test_loss += loss_val.item()
        count += output.size(0)

        _, pred1 = output.max(dim=1)
        ag = (pred1 == target)
        am = ag.sum()
        pred += am.item()

    test_loss = float(test_loss)/count
    accuracy = float(pred)/count
    return test_loss, accuracy

def train_one_epoch(device, model, train_loader, optimizer):
    model.train()
    train_loss = 0.0
    pred = 0.0
    count = 0
    for i, data in enumerate(tqdm(train_loader)):
        points, target = data
        target = target[:,0]

        points = points.to(device)
        target = target.to(device)

        output = model(points)
        loss_val = torch.nn.functional.nll_loss(
            torch.nn.functional.log_softmax(output, dim=1), target, size_average=False)

        # forward + backward + optimize
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        train_loss += loss_val.item()
        count += output.size(0)

        _, pred1 = output.max(dim=1)
        ag = (pred1 == target)
        am = ag.sum()
        pred += am.item()

    train_loss = float(train_loss)/count
    accuracy = float(pred)/count
    return train_loss, accuracy


def main():
    # Init WANDB for metadata saving
    wandb.init(project='skeleton-to-age', entity='lotemn102', name=config.RUN_NAME)

    # Save the parameters to WANDB
    wand_config = wandb.config
    wand_config.learning_rate = config.LEARNING_RATE
    wand_config.batch_size = config.BATCH_SIZE
    wand_config.num_workers = config.NUM_WORKERS
    wand_config.embedding_dims = config.EMBEDDING_DIMS
    wand_config.optimizer = config.OPTIMIZER
    wand_config.num_epochs = config.NUM_EPOCHS

    train_dict = {}
    test_dict = {}

    x_train, x_test, y_train, y_test = read()
    train_dict['pcs'] = x_train
    train_dict['labels'] = y_train.reshape(-1, 1)
    test_dict['pcs'] = x_test
    test_dict['labels'] = y_test.reshape(-1, 1)
    train_set = UserData('classification', train_dict)
    test_set = UserData('classification', test_dict)

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True,
                              num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE, shuffle=False, drop_last=False,
                             num_workers=config.NUM_WORKERS)

    # Create PointNet Model.
    ptnet = PointNet(emb_dims=config.EMBEDDING_DIMS, use_bn=True)
    model = Classifier(feature_model=ptnet, num_classes=2)
    model.to(config.DEVICE)

    # Watch model
    wandb.watch(model)

    checkpoint = None

    if config.LOAD_CHECKPOINT:
        assert os.path.isfile(config.CHECKPOINT_PATH)
        checkpoint = torch.load(config.CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model'])
        print("Starting training from checkpoint {n}.".format(n=config.CHECKPOINT_PATH))

    learnable_params = filter(lambda p: p.requires_grad, model.parameters())
    if config.OPTIMIZER == 'Adam':
        optimizer = torch.optim.Adam(learnable_params)
    else:
        optimizer = torch.optim.SGD(learnable_params, lr=config.LEARNING_RATE)

    if checkpoint is not None:
        min_loss = checkpoint['min_loss']
        optimizer.load_state_dict(checkpoint['optimizer'])

    best_test_loss = np.inf

    for epoch in range(config.NUM_EPOCHS):
        train_loss, train_accuracy = train_one_epoch(config.DEVICE, model, train_loader, optimizer)
        test_loss, test_accuracy = test_one_epoch(config.DEVICE, model, test_loader)

        # Watch losses
        wandb.log({"train loss": train_loss, "test loss": test_loss, "train accuracy": train_accuracy,
                   "test accuracy": test_accuracy})

        if test_loss < best_test_loss:
            best_test_loss = test_loss

            snap = {'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'min_loss': best_test_loss,
                    'optimizer': optimizer.state_dict(), }
            torch.save(snap, 'checkpoints/{run_name}.pth.tar'.format(run_name=config.RUN_NAME))
            print('Checkpoint saved.')

        print('EPOCH:: %d, Training Loss: %f, Testing Loss: %f, Best Loss: %f' % (epoch + 1, train_loss, test_loss, best_test_loss))
        print('EPOCH:: %d, Training Accuracy: %f, Testing Accuracy: %f' % (epoch + 1, train_accuracy, test_accuracy))

if __name__ == "__main__":
    main()