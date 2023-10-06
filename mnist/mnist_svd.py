from __future__ import print_function
import argparse
import torch
#import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
from torchvision import datasets, transforms
#from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
#import time
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1 + torch.exp(-z))

def der_sigmoid(z):
    num = torch.exp(-z)
    den = (1 + num)**2
    return num/den

def estimate_activations(args, weights, ground_truth_list, num_classes, batch_idx, add_bias = True, method = "Backprop"):
    unique_grouth_truths = torch.unique(ground_truth_list)
    inputs_approx = []
    Z = {}
    if method == "Backprop":
        input_dim = weights[0].shape[0]
        if add_bias:
            input_dim -= 1
        sample_input = torch.ones(1,input_dim)
        #sample_input = torch.rand(1,weights[0].shape[0])
        A_sample = forward_pass(sample_input, weights, return_all = True, add_bias=add_bias)
        num_layers = len(weights)
        for y in unique_grouth_truths:
            # Get one hot vector of the ground truth
            Y = F.one_hot(y, num_classes=num_classes).reshape(1,-1)
            y_value = y.item()
            #print(f"Calculating for y = [{y_value}] =====>")
            Z[y_value] = {}
            error = A_sample[-1].reshape(1,-1) - Y # 1x10
            A_out = Y # 1x10
            old_delta = error
            #new_weights = [0]*num_layers
            for layer in reversed(range(num_layers)):
                # Iterates from last layer to first layer
                Z[y_value][layer] = sigmoid_inverse(A_out).T # 10x1
                # Calculate delta
                new_delta = old_delta * der_sigmoid(A_sample[layer+1]) # 1x10
                # ===
                #new_w_delta = new_delta.T @ A_sample[layer] # 10x1|1x128: 10x128
                #new_weights[layer] = weights[layer] - new_w_delta.T # 128x10
                # ===
                if add_bias:
                    new_delta = weights[layer][:-1,:] @ new_delta.T # 128x10|10x1: 128x1
                else:
                    new_delta = weights[layer] @ new_delta.T # 128x10|10x1: 128x1
                old_delta = new_delta.T # 1x128
                A_out = A_sample[layer] - old_delta # 1x128
                
                # if layer == 0:
                #     Z_1 = Z[y_value][0].T
                #     A_1 = torch.sigmoid(Z_1)
                #     Z_2 = A_1@weights[1]
                #     A_2 = torch.sigmoid(Z_2)
                #     Z_3 = A_2@weights[2]
                #     A_3 = torch.sigmoid(Z_3)
                #     print(f"Prediction of weights [{y_value}] --> [{torch.round(A_3*100)/100}]")
                #     Z_2 = Z[y_value][1].T
                #     A_2 = torch.sigmoid(Z_2)
                #     Z_3 = A_2@weights[2]
                #     A_3 = torch.sigmoid(Z_3)
                #     print(f"Prediction of weights [{y_value}] --> [{torch.round(A_3*100)/100}]")
                #     print(Z[y_value].keys())
                #     Z_3 = Z[y_value][2].T
                #     A_3 = torch.sigmoid(Z_3)
                #     print(f"Prediction of weights [{y_value}] --> [{torch.round(A_3*100)/100}]")

            inputs_approx.append(A_out)        
    elif method == "weight_compare":
        for y in unique_grouth_truths:
            A_out = F.one_hot(y, num_classes=num_classes)
            y_value = y.item()
            Z[y_value] = {}
            # TODO: Vectorize this for loop. Calculate Z for multiple y values at once.
            for j in reversed(range(len(weights))):
                # Iterates from last layer to first layer
                Z_out = sigmoid_inverse(A_out)
                # Save Z to a serperate list
                Z[y_value][j] = Z_out
                #print(f"Layer {j} => Z.shape: {Z_out.shape}")
                Z_size = Z_out.shape[0]
                A = torch.zeros(weights[j].shape[0])
                #print(f"layer[{j}] ---> {priority_list}")
                for k in range(Z_size):
                    # Iterates through each node of a layer
                    if Z_out[k] > 0:
                        # Maximize Z
                        # Assign higher weight to priority Z values
                        A_k_approx = weights[j][:,k] > 0
                    else:
                        # Minimize Z
                        A_k_approx = weights[j][:,k] < 0
                    A = A + A_k_approx
                # Save A as it's average from all the layers
                A_out = A/Z_size
                if add_bias:
                    A_out = A_out[:-1]
            inputs_approx.append(A_out)
        
    # Test approximated images for output
    for i in range(len(unique_grouth_truths)):
        print(f"Prediction of [{unique_grouth_truths[i]}] --> [{torch.round(forward_pass(inputs_approx[i], weights, add_bias=add_bias)*100)/100}]")
    # Clear all previous plots
    plt.close('all')
    fig = plt.figure(figsize=(8, 8))
    for i in range(len(unique_grouth_truths)):
        ax = fig.add_subplot(3, 4, i+1)
        ax.imshow(inputs_approx[i].view(28,28))
        ax.set_title(unique_grouth_truths[i])
    plt.savefig(f"approx_predictions/{batch_idx}{args.suffix}.png")
    #print(Z.keys())
    #print(Z[unique_grouth_truths[1].item()].keys())
    return Z

def calculate_weights(args, input, target, Z_est, principal_vecs, principal_vals, weights, add_bias = True):
    # Calculate weights using SVD
    num_layers = len(weights)
    batch_size = input.shape[0]
    Z = {}
    for layer in range(num_layers):
        out_dim = weights[layer].shape[1]
        Z[layer] = torch.zeros(out_dim,batch_size)
        for i in range(batch_size):
            # Calculate weights for each example
            # TODO: Vectorize this for loop. Try to calculate weights for multiple targets at once.
            Z[layer][:,i] = Z_est[target[i].item()][layer][:,0]

    A = input
    if add_bias:
        A = torch.concat((A,torch.ones(batch_size,1)),dim=1)
    for layer in range(num_layers):
        in_dim = weights[layer].shape[0]
        out_dim = weights[layer].shape[1]
        if add_bias:
            in_dim -= 1
        #print(f"A.shape: {A.shape}")
        if add_bias:
            batch_vecs = A.view(1,batch_size,in_dim+1).repeat(out_dim,1,1)
        else:
            batch_vecs = A.view(1,batch_size,in_dim).repeat(out_dim,1,1)
        batch_vals = Z[layer]
        #print(f"principal_vecs[{layer}]: {principal_vecs[layer].shape}")
        #print(f"principal_vals[{layer}]: {principal_vals[layer].shape}")
        #print(f"batch_vecs[{layer}]: {batch_vecs.shape}")
        #print(f"batch_vals[{layer}]: {batch_vals.shape}")
        p_vecs = principal_vecs[layer]
        if add_bias:
            p_vecs = torch.cat((p_vecs,torch.ones(out_dim,in_dim,1)),dim=2)
        all_vecs = torch.cat((p_vecs,batch_vecs),dim=1)
        all_vals = torch.cat((principal_vals[layer],batch_vals),dim=1)
        #print(f"all_vecs[{layer}]: {all_vecs.shape}")
        #print(f"all_vals[{layer}]: {all_vals.shape}")
        for i in tqdm(range(out_dim)):
            vecs = all_vecs[i,:,:]
            vals = all_vals[i,:]
            U, S, Vh = torch.linalg.svd(vecs, full_matrices=False)
            sigma_rec = 1/S
            weights[layer][:,i] = (Vh.T*sigma_rec)@U.T@vals
            # Calculate prinicipal vecs and values
            if add_bias:
                principal_vecs[layer][i,:,:] = (S.reshape(S.shape[0],1)*Vh)[:-1,:-1]
                principal_vals[layer][i,:] = (U.T@vals)[:-1]
            else:
                principal_vecs[layer][i,:,:] = S.reshape(S.shape[0],1)*Vh
                principal_vals[layer][i,:] = U.T@vals
        # Calculate A values for next layer
        Z_new = A@weights[layer]
        A = torch.sigmoid(Z_new)
        if add_bias:
            A = torch.concat((A,torch.ones(batch_size,1)),dim=1)
    
    #print(f"Final Output: {torch.argmax(A, dim=1)}")
    #print(f"Ground Truth: {target}")
    return weights,principal_vecs,principal_vals


def sigmoid_inverse(a:torch.Tensor):
    # Clamp the a values between 1*10^-6 and 1-1*10^-6
    return torch.special.logit(a, eps=1e-6) 

def train(args, train_loader, test_loader, add_bias = True):
    # Do not use gradient calculation
    torch.set_grad_enabled(False)
    # Set initial weights between -1 and 1
    # We have to transpose the weight matrix to be able to multiply matries with batches
    bias_dim = 0
    if add_bias:
        bias_dim = 1
    weights = []
    weights.append(2*torch.rand(784+bias_dim,512)-1)
    weights.append(2*torch.rand(512+bias_dim,128)-1)
    weights.append(2*torch.rand(128+bias_dim,10)-1)
    
    num_classes=weights[-1].shape[1]
    # Initialize eqautions for every node
    # Eg: For one node, they would be of shape (w.shape[0],w.shape[0])
    # Remember that shape of z is always (batch_size,w.shape[0],1)
    principal_vecs = []
    principal_vals = []
    train_acc_list = []
    imm_train_acc_list = []
    test_acc_list = []
    num_images_trained = []
    for layer in range(len(weights)):
        in_dim = weights[layer].shape[0]
        out_dim = weights[layer].shape[1]
        if add_bias:
            in_dim -= 1
        principal_vecs.append(torch.zeros(out_dim,in_dim,in_dim))
        principal_vals.append(torch.zeros(out_dim,in_dim))
    
    input_dim = weights[0].shape[0]
    if add_bias:
        input_dim -= 1
    old_data = torch.zeros(0,input_dim)
    old_target = torch.zeros(0)
    
    for epoch in range(1, args.epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data = torch.flatten(data, start_dim=1)
            Z_est = estimate_activations(args, weights, target, num_classes, batch_idx, add_bias=add_bias)
            weights,principal_vecs,principal_vals = calculate_weights(args, data, target, Z_est, principal_vecs, principal_vals, weights, add_bias=add_bias)
            old_data = torch.cat([old_data, data],dim=0)
            #print(f"old_target.shape: {old_target.shape}, target.shape: {target.shape}")
            old_target = torch.cat([old_target, target],dim=0)
            if (old_data.shape[0] > 0) and (old_target.shape[0] > 0):
                pred = torch.argmax(forward_pass(old_data, weights, add_bias=add_bias), dim=1)
                imm_pred = torch.argmax(forward_pass(data, weights, add_bias=add_bias), dim=1)
                acc = 100 * torch.sum(pred == old_target)/old_data.shape[0]
                imm_acc = 100 * torch.sum(imm_pred == target)/data.shape[0]
                print(f"Old Data Accuracy: {acc}% | No. of examples: {old_data.shape[0]}/{len(train_loader)*data.shape[0]}")
                test_acc = test(test_loader,weights,add_bias=add_bias)
                # Save accuracies to a list
                train_acc_list.append(acc)
                imm_train_acc_list.append(imm_acc)
                test_acc_list.append(test_acc)
                num_images_trained.append(((epoch-1)*len(train_loader))+batch_idx)
                # Clear all previous plots
                plt.close('all')
                plt.plot(num_images_trained, train_acc_list, marker='o', label='Old Train Acc')
                plt.plot(num_images_trained, imm_train_acc_list, marker='o', label='Immediate Train Acc')
                plt.plot(num_images_trained, test_acc_list, marker='o', label='Test Acc')
                plt.legend()
                plt.savefig(f"acc{args.suffix}.png")

            if args.dry_run:
                break
        
        if args.dry_run:
            break
    
    return weights,principal_vecs,principal_vals

def forward_pass(input, weights, return_all = False, add_bias = True):
    all_A = [input]
    A = input
    for layer in range(len(weights)):
        if add_bias:
            A = torch.concat((A,torch.ones(input.shape[0],1)),dim=1)
        Z = A@weights[layer]
        A = torch.sigmoid(Z)
        all_A.append(A)
    
    if return_all:
        return all_A
    return all_A[-1]

def test(test_loader,weights,add_bias=True):
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = torch.flatten(data, start_dim=1)

            pred = forward_pass(data,weights,add_bias=add_bias).argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = 100. * correct / len(test_loader.dataset)
    print('\nTest Set Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset), test_acc))
    return test_acc

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
    parser.add_argument('--no-bias', action='store_true', default=False,
                        help='Run model with no bias parameter')
    parser.add_argument('--suffix', type=str, default="", metavar='N',
                        help='Add a suffix to results do distinguish this run from other runs')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)
    plt.set_cmap('gray')

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

    weights,principal_vecs,principal_vals = train(args, train_loader, test_loader, add_bias=(not args.no_bias))

if __name__ == '__main__':
    main()