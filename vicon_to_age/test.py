import open3d as o3d
import torch
from torch.utils.data import DataLoader

import config
from learning3d.data_utils import UserData
from learning3d.models import PointNet
from learning3d.models import Classifier
from data_reader import read

def display_open3d(template):
    template_ = o3d.geometry.PointCloud()
    template_.points = o3d.utility.Vector3dVector(template)
    o3d.visualization.draw_geometries([template_])

def test_one_epoch(device, model, test_loader, testset):
    model.eval()
    test_loss = 0.0
    pred  = 0.0
    count = 0
    for i, data in enumerate(test_loader):
        points, target = data
        target = target[:,0]

        points = points.to(device)
        target = target.to(device)

        output = model(points)
        loss_val = torch.nn.functional.nll_loss(
            torch.nn.functional.log_softmax(output, dim=1), target, size_average=False)
        label = int(testset.data_class.labels[i])
        prediction = torch.argmax(output[0]).item()
        # print("Ground Truth Label: ", label)
        # print("Predicted Label:    ", prediction)
        # print("------------------------------------------------------")

        if label != prediction:
            print("Ground Truth Label: ", label)
            print("Predicted Label:    ", prediction)
            print("------------------------------------------------------")

        #display_open3d(points.detach().cpu().numpy()[0])

        test_loss += loss_val.item()
        count += output.size(0)

        _, pred1 = output.max(dim=1)
        ag = (pred1 == target)
        am = ag.sum()
        pred += am.item()

    test_loss = float(test_loss)/count
    accuracy = float(pred)/count
    return test_loss, accuracy


def main():
    test_dict = {}

    x_train, x_test, y_train, y_test = read()
    test_dict['pcs'] = x_test
    test_dict['labels'] = y_test.reshape(-1, 1)
    test_set = UserData('classification', test_dict)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, drop_last=False,
                             num_workers=config.NUM_WORKERS)

    # Create PointNet Model.
    ptnet = PointNet(emb_dims=config.EMBEDDING_DIMS, use_bn=True)
    model = Classifier(feature_model=ptnet, num_classes=2)
    checkpoint = torch.load(config.CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model'])

    loss, accuracy = test_one_epoch(config.DEVICE, model, test_loader, test_set)
    print("Test loss is {loss}, test accuracy is {acc}".format(loss=loss, acc=accuracy))


if __name__ == '__main__':
    main()