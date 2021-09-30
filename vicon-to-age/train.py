import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from learning3d.data_utils import UserData
from learning3d.models import PointNet
from learning3d.models import Classifier
from data_reader import read


BATCH_SIZE = 32
NUM_WORKERS = 2
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
EMBEDDING_DIMS = 1024
CHECKPOINT_PATH = None
OPTIMIZER = 'Adam'
NUM_EPOCHS = 200
EXP_NAME = 'run_01'


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
    train_dict = {}
    test_dict = {}

    x_train, x_test, y_train, y_test = read()
    train_dict['pcs'] = x_train
    train_dict['labels'] = y_train.reshape(-1, 1)
    test_dict['pcs'] = x_test
    test_dict['labels'] = y_test.reshape(-1, 1)
    train_set = UserData('classification', train_dict)
    test_set = UserData('classification', test_dict)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                              num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
                             num_workers=NUM_WORKERS)

    # Create PointNet Model.
    ptnet = PointNet(emb_dims=EMBEDDING_DIMS, use_bn=True)
    model = Classifier(feature_model=ptnet, num_classes=100)
    model.to(DEVICE)

    checkpoint = None

    if CHECKPOINT_PATH:
        assert os.path.isfile(CHECKPOINT_PATH)
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model'])

    learnable_params = filter(lambda p: p.requires_grad, model.parameters())
    if OPTIMIZER == 'Adam':
        optimizer = torch.optim.Adam(learnable_params)
    else:
        optimizer = torch.optim.SGD(learnable_params, lr=0.1)

    if checkpoint is not None:
        min_loss = checkpoint['min_loss']
        optimizer.load_state_dict(checkpoint['optimizer'])

    best_test_loss = np.inf

    for epoch in range(NUM_EPOCHS):
        train_loss, train_accuracy = train_one_epoch(DEVICE, model, train_loader, optimizer)
        test_loss, test_accuracy = test_one_epoch(DEVICE, model, test_loader)

        # if test_loss < best_test_loss:
        #     best_test_loss = test_loss
        #     snap = {'epoch': epoch + 1,
        #             'model': model.state_dict(),
        #             'min_loss': best_test_loss,
        #             'optimizer': optimizer.state_dict(), }
        #     torch.save(snap, 'checkpoints/%s/models/best_model_snap.t7' % (EXP_NAME))
        #     torch.save(model.state_dict(), 'checkpoints/%s/models/best_model.t7' % (EXP_NAME))
        #     torch.save(model.feature_model.state_dict(), 'checkpoints/%s/models/best_ptnet_model.t7' % (EXP_NAME))

        # torch.save(snap, 'checkpoints/%s/models/model_snap.t7' % (EXP_NAME))
        # torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % (EXP_NAME))
        # torch.save(model.feature_model.state_dict(), 'checkpoints/%s/models/ptnet_model.t7' % (EXP_NAME))

        print('EPOCH:: %d, Traininig Loss: %f, Testing Loss: %f, Best Loss: %f' % (epoch + 1, train_loss, test_loss, best_test_loss))
        print('EPOCH:: %d, Traininig Accuracy: %f, Testing Accuracy: %f' % (epoch + 1, train_accuracy, test_accuracy))

if __name__ == "__main__":
    main()