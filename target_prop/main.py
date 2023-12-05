import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import pickle
import datetime
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Setting up command line arguments with a more structured approach
def configure_arg_parser():
    parser = argparse.ArgumentParser(description='Configuration for Target Propagation Neural Network')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (default: 32)')
    parser.add_argument('--test_batch_size', type=int, default=1000, help='Batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs for training (default: 1)')
    parser.add_argument('--lr_tab', nargs='+', type=float, default=[0.05, 0.01], help='Learning rates for different layers (default: [0.05, 0.01])')
    parser.add_argument('--size_tab', nargs='+', type=int, default=[784, 512, 10], help='Sizes of layers in the network (default: [784, 512, 10])')
    parser.add_argument('--action', choices=['train', 'test'], default='test', help='Action to perform: train or test (default: test)')
    parser.add_argument('--sigma', type=float, default=0.01, help='Standard deviation of noise for feedback weights training (default: 0.01)')
    parser.add_argument('--lr_target', type=float, default=0.01, help='Learning rate for last layer target computation (default: 0.01)')
    parser.add_argument('--device', type=int, default=0, help='CUDA device selection (default: 0, -1 for CPU)')
    parser.add_argument('--SDTP', action='store_true', help='Enable Simplified Target Propagation (default: False)')

    return parser.parse_args()

args = configure_arg_parser()

batch_size = args.batch_size
batch_size_test = args.test_batch_size

class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


mnist_transforms=[torchvision.transforms.ToTensor(),ReshapeTransform((-1,))]

train_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST(root='./data', train=True, download=True,
                     transform=torchvision.transforms.Compose(mnist_transforms)),
                     #target_transform=ReshapeTransformTarget(10)),
batch_size = args.batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST(root='./data', train=False, download=True,
                     transform=torchvision.transforms.Compose(mnist_transforms)),
                     #target_transform=ReshapeTransformTarget(10)),
batch_size = args.test_batch_size, shuffle=True)


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        self.size_tab = args.size_tab

        forward_weight = nn.ModuleList([])
        backward_weight = nn.ModuleList([])

        for i in range(len(self.size_tab) - 1):
            forward_weight.append(nn.Linear(self.size_tab[i], self.size_tab[i + 1]))
        for i in range(len(self.size_tab) - 2):
            backward_weight.append(nn.Linear(self.size_tab[-1 - i], self.size_tab[-2 - i]))

        self.logsoft = nn.LogSoftmax(dim=1)
        self.forward_weight = forward_weight
        self.backward_weight = backward_weight
        self.lr_target = args.lr_target
        self.SDTP = args.SDTP
        self.sigma = args.sigma

    def forward(self, input_tensor):
        activations = [input_tensor]
        
        # Apply tanh activation to each layer except the last
        for layer in self.forward_weight[:-1]:
            activations.append(torch.tanh(layer(activations[-1])))

        # Apply log softmax activation to the output of the last layer
        activations.append(torch.exp(self.logsoft(self.forward_weight[-1](activations[-1]))))

        return activations


    def computeTargets(self, activations, targets, loss_criterion):
        target_list = []

        if self.SDTP:
            # Target for the output layer in SDTP mode
            target_list.append(F.one_hot(targets, num_classes=self.layer_sizes[-1]).float())

            # Compute targets for the lower layers
            for index in range(len(activations) - 2):
                current_target = activations[index + 1] - torch.tanh(self.backward_weight[index](activations[index])) + torch.tanh(self.backward_weight[index](target_list[index]))
                target_list.append(current_target)
        else:
            # In non-SDTP mode, no target for the output layer
            target_list.append(None)

            # Target for the penultimate layer
            loss = loss_criterion(activations[0].float(), targets)
            initial_gradient = torch.ones(targets.size(0), dtype=torch.float, device=targets.device, requires_grad=True)
            gradient = torch.autograd.grad(loss, activations[1], grad_outputs=initial_gradient, retain_graph=True)[0]
            penultimate_target = activations[1] - self.lr_target * gradient
            target_list.append(penultimate_target)

            # Compute targets for the lower layers
            for index in range(1, len(activations) - 2):
                current_target = activations[index + 1] - torch.tanh(self.backward_weight[index](activations[index])) + torch.tanh(self.backward_weight[index](target_list[index]))
                target_list.append(current_target)

        return target_list


    def reconstruct(self, activation_sequence, layer_index):
        # Apply forward and backward operations to reconstruct the data
        forward_pass = torch.tanh(self.forward_weight[-layer_index](activation_sequence))
        reconstructed = torch.tanh(self.backward_weight[layer_index - 1](forward_pass))

        return reconstructed      

if __name__ == '__main__':
    if args.device >= 0:
        device = torch.device("cuda:"+str(args.device))

    nlll = nn.NLLLoss(reduction='none')
    mse = torch.nn.MSELoss(reduction='sum')
    net = Net(args)
    net.to(device)


    if args.action == 'train':
        net.train()

        #build optimizers for forward weights
        optim_wf_param = []
        for i in range(len(net.forward_weight) - 1):
            optim_wf_param.append({'params':net.forward_weight[i].parameters(), 'lr': args.lr_tab[0]})
        optimizer_wf = torch.optim.SGD(optim_wf_param)

        #build optimizers for the last forward weight
        optimizer_wf_top = torch.optim.SGD([{'params':net.forward_weight[-1].parameters(), 'lr': args.lr_tab[0]}])

        #build optimizers for backward weights
        optim_wb_param = []
        for i in range(len(net.backward_weight)):
            optim_wb_param.append({'params':net.backward_weight[i].parameters(), 'lr': args.lr_tab[0]})
        optimizer_wb = torch.optim.SGD(optim_wb_param)          

        #start training
        train_acc = []
        train_std = []
        for epoch in range(1, args.epochs + 1):
            correct = []
            print('Epoch {}'.format(epoch))
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(device), targets.to(device)

                #forward pass
                s = net(data)

                #compute targets
                s.reverse()
                t = net.computeTargets(s, targets, nlll)

                #train backward weights
                for i in range(len(s) - 2):
                    #generate corrupted data
                    s_corrupt = s[i + 1] + net.sigma*torch.randn_like(s[i + 1])

                    #reconstruct the data
                    r = net.reconstruct(s_corrupt, i + 1)

                    #update the backward weights
                    loss_wb = (1/(2*data.size(0)))*mse(s[i + 1], r)
                    optimizer_wb.zero_grad()
                    loss_wb.backward(retain_graph = True)
                    optimizer_wb.step()
                
                #train forward weights
                s.reverse()
                t.reverse()

                for i in range(len(s) - 2):
                    loss_wf = (1/(2*data.size(0)))*mse(s[i + 1], t[i])
                    optimizer_wf.zero_grad()
                    loss_wf.backward(retain_graph = True)
                    optimizer_wf.step()
                                    
                #train the top forward weights
                loss_wf_top = nlll(s[-1].float(), targets).mean()  
                optimizer_wf_top.zero_grad()
                loss_wf_top.backward(retain_graph = True)
                optimizer_wf_top.step()

                #compute prediction error
                pred = s[-1].data.max(1, keepdim=True)[1]
                correct.append(pred.eq(targets.data.view_as(pred)).cpu().sum())
                

            print('\nAverage Training Error Rate: {:.2f}% ({}/{})\n'.format(
                    100*(len(train_loader.dataset)- np.sum(correct)  )/ len(train_loader.dataset), 
                    len(train_loader.dataset)-np.sum(correct) , len(train_loader.dataset)))

            train_acc.append(100*(len(train_loader.dataset)- np.sum(correct) )/ len(train_loader.dataset))
            train_std.append(np.std(correct))

        plt.figure()
        epochs = [i for i in range(args.epochs)]
        plt.errorbar(epochs, train_acc, yerr=train_std, fmt='-o', ecolor='blue', capsize=5, label="diff_target_prop_loss")
        plt.xlabel("Epochs")
        plt.ylabel("Error Rate")
        plt.title("Average Training Error Rate")
        plt.legend()
        plt.savefig("result.png")


    if args.action == 'test':
        _, (data, target) = next(enumerate(train_loader))
        data, target = data.to(device), target.to(device)

        s = net(data)

        #check target computation        
        s.reverse()
        t = net.computeTargets(s, target, nlll)                
        for i in t:
            if i is None:
                print('No target')
            else:
                print(i.size())
        
        

















