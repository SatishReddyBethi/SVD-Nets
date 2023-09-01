from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import time

def train(args, train_loader, epoch):
    # Set initial weights
    # We have to transpose the weight matrix to be able to multiply matries with batches
    w1 = torch.rand(512,784).T
    w2 = torch.rand(128,512).T
    w3 = torch.rand(10,128).T
    # TODO: Call equations as vectors. Eg: Principal vectors
    # Initialize eqautions for every node
    # Eg: For one node, they would be of shape (w.shape[0],w.shape[0])
    # Remember that shape of z is always (batch_size,w.shape[0],1)
    principal_eqns1 = torch.zeros(w1.shape[1],w1.shape[0],w1.shape[0])
    principal_eqns2 = torch.zeros(w2.shape[1],w2.shape[0],w2.shape[0])
    principal_eqns3 = torch.zeros(w3.shape[1],w3.shape[0],w3.shape[0])
    principal_vals1 = torch.zeros(w1.shape[1],w1.shape[0])
    principal_vals2 = torch.zeros(w2.shape[1],w2.shape[0])
    principal_vals3 = torch.zeros(w3.shape[1],w3.shape[0])
    for batch_idx, (data, target) in enumerate(train_loader):
        data = torch.flatten(data, start_dim=1)
        # Calculate forward pass and get equations for all the nodes
        # input
        a0 = data
        print(f"a0_shape: {a0.shape}")
        # layer 1
        z1 = a0@w1
        a1 = sigmoid(z1)
        print(f"a1_shape: {a1.shape}")
        # layer 2
        z2 = a1@w2
        a2 = sigmoid(z2)
        print(f"a2_shape: {a2.shape}")
        # layer 3
        z3 = a2@w3
        a3 = sigmoid(z3)
        print(f"a3_shape: {a3.shape}")
        print("==========================")
        # Get new equations from all layers (batch_size,a.shape[1],z.shape[1]+z.shape[2])
        # a.shape[1] is the size of output a (batch_size, a.shape[1])
        layer1_eqns = a0
        layer1_vals = z1.T
        layer2_eqns = a1
        layer2_vals = z2.T
        layer3_eqns = a2
        layer3_vals = z3.T
        # Combine the new equations with principal equations
        # TODO: Test torch.Tensor.repeat instead of multiplying by torch.ones
        layer1_eqns = torch.ones(w1.shape[1],args.batch_size,w1.shape[0])*layer1_eqns.view(1,args.batch_size,w1.shape[0])
        layer2_eqns = torch.ones(w2.shape[1],args.batch_size,w2.shape[0])*layer2_eqns.view(1,args.batch_size,w2.shape[0])
        layer3_eqns = torch.ones(w3.shape[1],args.batch_size,w3.shape[0])*layer3_eqns.view(1,args.batch_size,w3.shape[0])
        print(f"principal_eqns1: {principal_eqns1.shape}")
        print(f"layer1_eqns: {layer1_eqns.shape}")
        all_eqns1 = torch.cat((principal_eqns1,layer1_eqns),1)
        all_eqns2 = torch.cat((principal_eqns2,layer2_eqns),1)
        all_eqns3 = torch.cat((principal_eqns3,layer3_eqns),1)
        print(f"all_eqns1: {all_eqns1.shape}\nall_eqns2: {all_eqns2.shape}\nall_eqns3: {all_eqns3.shape}")
        print("==========================")
        # Combine the new values with principal values
        #layer1_vals = torch.ones(w1.shape[1],args.batch_size,w1.shape[0])*layer1_vals.view(1,args.batch_size,w1.shape[0])
        #layer2_vals = torch.ones(w2.shape[1],args.batch_size,w2.shape[0])*layer2_eqns.view(1,args.batch_size,w2.shape[0])
        #layer3_vals = torch.ones(w3.shape[1],args.batch_size,w3.shape[0])*layer3_eqns.view(1,args.batch_size,w3.shape[0])
        print(f"principal_vals1: {principal_vals1.shape}")
        print(f"layer1_vals: {layer1_vals.shape}")
        all_vals1 = torch.cat((principal_vals1,layer1_vals),1)
        all_vals2 = torch.cat((principal_vals2,layer2_vals),1)
        all_vals3 = torch.cat((principal_vals3,layer3_vals),1)
        print(f"all_vals1: {all_vals1.shape}\nall_vals2: {all_vals2.shape}\nall_vals3: {all_vals3.shape}")
        print("==========================")
        # Calculate all the parameters from equations using SVD
        # Iterative implementation of SVD
        st = time.time()
        for i in tqdm(range(all_eqns1.shape[0])):
            eqns = all_eqns1[i,:,:]
            vals = all_vals1[i,:]
            U, S, Vh = torch.linalg.svd(eqns, full_matrices=False)
            sigma_rec = 1/S
            w1[:,i] = (Vh.T*sigma_rec)@U.T@vals
            # Calculate prinicipal eqns and values
            principal_eqns1[i,:,:] = S*Vh
            principal_vals1[i,:] = U.T@vals
        
        print(f"Iterative SVD Execution time: {time.time() - st} seconds")

        # Vectorized implementation of SVD
        st = time.time()
        U, S, Vh = torch.linalg.svd(all_eqns1, full_matrices=False)
        print(S[0,100])
        S_new = S.reshape(all_eqns1.shape[0],1,S.shape[1])
        Sigma_rec = 1/S_new
        values = all_vals1.reshape(all_eqns1.shape[0],-1,1)
        w1 = (Vh.permute(0,2,1) * Sigma_rec)@U.permute(0,2,1)@values
        principal_eqns1 = S_new*Vh
        principal_vals1 = U.permute(0,2,1)@values
        print('Vectorized SVD Execution time:', time.time() - st, 'seconds')
        # Replace the old parameters with the newly calculated parameters and principal equations
        if args.dry_run:
            break

def sigmoid(x):
    return 1/(1+torch.exp(-x))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    for epoch in range(1, args.epochs + 1):
        train(args, train_loader, epoch)
        if args.dry_run:
            break

if __name__ == '__main__':
    main()