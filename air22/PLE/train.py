import torch
import torch.nn as nn
import numpy as np
from ple import PLE
from datasets import build_tensor_dataset, data_preparation
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")


def test(dataloader, loss_fn):
    task1_pred_all, task2_pred_all, task1_label_all, task2_label_all = [], [], [], []
    model.eval()
    with torch.no_grad():
        for data, label in dataloader:
            data, label = data.to(device), label.to(device)
            pred = model(data)
            task1_label, task2_label = label[:, 0], label[:, 1]
            task1_pred, task2_pred = pred[0], pred[1]

            loss1 = loss_fn(task1_pred.view(-1), task1_label)
            loss2 = loss_fn(task2_pred.view(-1), task2_label)
            loss = loss1 + loss2

            task1_pred_all.extend(list(task1_pred.cpu()))
            task2_pred_all.extend(list(task2_pred.cpu()))
            task1_label_all.extend(list(task1_label.cpu()))
            task2_label_all.extend(list(task2_label.cpu()))

    task1_pred_all = np.array(task1_pred_all)
    task2_pred_all = np.array(task2_pred_all)
    task1_label_all = np.array(task1_label_all)
    task2_label_all = np.array(task2_label_all)
    auc_1 = roc_auc_score(task1_label_all, task1_pred_all)
    auc_2 = roc_auc_score(task2_label_all, task2_pred_all)
    return auc_1, auc_2


EPOCHS = 100
BATCH_SIZE = 1024
LR = 1e-4

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_data, train_label, valid_data, valid_label, test_data, test_label = data_preparation()
train_loader = torch.utils.data.DataLoader(dataset=build_tensor_dataset(train_data, train_label),
                                           batch_size=BATCH_SIZE, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=build_tensor_dataset(valid_data, valid_label),
                                           batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(dataset=build_tensor_dataset(test_data, test_label),
                                          batch_size=BATCH_SIZE)
model = PLE(num_layers=2, dim_x=499, dim_experts_out=[256, 128], num_experts=[8, 8, 8], num_tasks=2, dim_tower=[128, 64, 1])

print(f"Using {device} device")
loss_fn = nn.BCELoss(reduction='mean')
model = model.to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
losses = []
val_loss = []
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = []
    print("Epoch: {}/{}".format(epoch, EPOCHS))
    for data, label in train_loader:
        data, label = data.to(device), label.to(device)
        pred = model(data)

        task1_label, task2_label = label[:, 0], label[:, 1]
        task1_pred, task2_pred = pred[0], pred[1]

        optimizer.zero_grad()
        loss1 = loss_fn(task1_pred.view(-1), task1_label)
        loss2 = loss_fn(task2_pred.view(-1), task2_label)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
    losses.append(np.mean(epoch_loss))

    auc1, auc2 = test(valid_loader, loss_fn)
    print('train loss: {:.5f}, val task1 auc: {:.5f}, val task2 auc: {:.3f}'.format(np.mean(epoch_loss), auc1, auc2))

    if epoch % 10 == 0:
        torch.save(model.state_dict(), 'ple_model_epoch_%d.pth' % epoch)
        print('Save PLE model.')

auc1, auc2 = test(test_loader, loss_fn)
print('test task1 auc: {:.3f}, test task2 auc: {:.3f}'.format(auc1, auc2))
torch.save(model.state_dict(), 'ple_model_epoch_%d.pth' % epoch)
print('Save PLE model.')
