import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from lib import fa_linear
from lib import linear
import os
from angle import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


BATCH_SIZE = 32

# load mnist dataset
train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))
                                         ])),
                          batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(datasets.MNIST('./data', train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                        ])),
                         batch_size=BATCH_SIZE, shuffle=True)

# load feedforward dfa model
model_fa = fa_linear.LinearFANetwork(in_features=784, num_layers=2, num_hidden_list=[1000, 10]).to(device)

# load reference linear model
model_bp = linear.LinearNetwork(in_features=784, num_layers=2, num_hidden_list=[1000, 10]).to(device)

# optimizers
optimizer_fa = torch.optim.SGD(model_fa.parameters(),
                            lr=1e-4, momentum=0.9, weight_decay=0.001, nesterov=True)
optimizer_bp = torch.optim.SGD(model_bp.parameters(),
                            lr=1e-4, momentum=0.9, weight_decay=0.001, nesterov=True)

loss_crossentropy = torch.nn.CrossEntropyLoss()

# make log file
results_path = 'bp_vs_fa_'
logger_train = open(results_path + 'train_log.txt', 'w')

# train loop
epochs = 50
all_fa_loss = []
all_bp_loss = []
all_angle = []
all_angle_hidden = []

fa_step_loss = []
bp_step_loss = []
fa_step_std = []
bp_step_std = []

for epoch in range(epochs):
    fa_loss = []
    bp_loss = []
    angle = []
    angle_hidden = []
    for idx_batch, (inputs, targets) in enumerate(train_loader):
        # flatten the inputs from square image to 1d vector
        inputs = inputs.view(BATCH_SIZE, -1)
        # wrap them into varaibles
        inputs, targets = Variable(inputs), Variable(targets)
        # get outputs from the model
        hidden_fa = model_fa(inputs.to(device))[0]
        outputs_fa = model_fa(inputs.to(device))[1]
        hidden_bp = model_bp(inputs.to(device))[0]
        outputs_bp = model_bp(inputs.to(device))[1]
        # calculate loss
        loss_fa = loss_crossentropy(outputs_fa, targets.to(device))
        loss_bp = loss_crossentropy(outputs_bp, targets.to(device))

        # angle calculation
        # outputs_fa_copy = outputs_fa.copy().detach().numpy()
        sig_fa_copy = torch.tensor(sigmoid_derivative(hidden_fa))
        # outputs_bp_copy = outputs_bp.copy().detach().numpy()
        sig_bp_copy = torch.tensor(sigmoid_derivative(hidden_bp))

        hyp_angle = angle_between_matrices(outputs_fa, outputs_bp)
        # hyp_angle2 = angle_between_matrices(hidden_fa, hidden_bp)

        angle.append(hyp_angle)
        # angle_hidden.append(hyp_angle2)

        model_fa.zero_grad()
        loss_fa.backward()
        optimizer_fa.step()

        model_bp.zero_grad()
        loss_bp.backward()
        optimizer_bp.step()
        
        fa_loss.append(loss_fa.item())
        bp_loss.append(loss_bp.item())
        fa_step_loss.append(loss_fa.item())
        bp_step_loss.append(loss_bp.item())

        if (idx_batch + 1) % 100 == 0:
            train_log = 'epoch ' + str(epoch) + ' step ' + str(idx_batch + 1) + \
                        ' loss_fa ' + str(loss_fa.item()) + ' loss_bp ' + str(loss_bp.item())
            print(train_log)
            logger_train.write(train_log + '\n')

    all_fa_loss.append(np.mean(fa_loss))
    all_bp_loss.append(np.mean(bp_loss))
    all_angle.append(np.mean(angle))
    fa_step_std.append(np.std(fa_loss))
    bp_step_std.append(np.std(bp_loss))
    # all_angle_hidden.append(np.mean(angle_hidden))

# figure 1
plt.figure()
epoch_lst = [i for i in range(epochs)]
plt.errorbar(epoch_lst, all_fa_loss, yerr=fa_step_std, fmt='-o', ecolor='blue', capsize=5, label="fa_loss")
plt.errorbar(epoch_lst, all_bp_loss, yerr=bp_step_std, fmt='-o', ecolor='red', capsize=5, label="bp_loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.title("Epoch Loss")
plt.savefig("2-layer-epoch.png")

# figure 2
plt.figure()
step_lst = [i for i in range(len(fa_step_loss))]
plt.plot(step_lst, fa_step_loss, label="fa_loss")
plt.plot(step_lst, bp_step_loss, label="bp_loss")
plt.legend()
plt.xlabel("Steps")
plt.ylabel("Losses")
plt.title("Step Loss")
plt.savefig("2-layer-step.png")

# figure 3
window = 5
averaged_angles = []
error_angles = []
ns = []
# all_angle = np.array(all_angle)
for i in range(len(all_angle) - window):
    ags = all_angle[i:i+window]
    averaged_angle = np.sum(ags)/float(window)
    error_angle = np.std(ags)
    averaged_angles.append(averaged_angle)
    error_angles.append(error_angle)
    ns.append(i)

plt.figure()
plt.errorbar(ns, averaged_angles, yerr=error_angles, color="#19AD1D", ecolor="#B5E6B5")
# plt.ylim([0,90])
plt.title("Angle")
plt.ylabel("Angle between BP and FA")
plt.grid(True)
plt.savefig('2-layer-angle.png')