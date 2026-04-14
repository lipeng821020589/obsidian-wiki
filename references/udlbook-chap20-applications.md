# Chap 20: 应用 (Applications)

> UDLbook 精读笔记
>
> **官方资源**: [GitHub Notebooks](https://github.com/udlbook/udlbook/tree/main/Notebooks/Chap20)

---

## Notebook 列表

- **随机数据**: `Chap20/20_1_Random_Data.ipynb`
- **全批量梯度下降**: `Chap20/20_2_Full_Batch_Gradient_Descent.ipynb`
- **彩票假说**: `Chap20/20_3_Lottery_Tickets.ipynb`
- **对抗攻击**: `Chap20/20_4_Adversarial_Attacks.ipynb`

---

## 内容

<a href="https://colab.research.google.com/github/udlbook/udlbook/blob/main/Notebooks/Chap20/20_1_Random_Data.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# **Notebook 20.1: Random Data**

This notebook investigates training the network with random data, as illustrated in figure 20.1.

Work through the cells below, running each cell in turn. In various places you will see the words "TODO". Follow the instructions at these places and make predictions about what is going to happen or write code to complete the functions.

Contact me at udlbookmail@gmail.com if you find any mistakes or have any suggestions.

```python

# Run this if you're in a Colab to install MNIST 1D repository
!pip install git+https://github.com/greydanus/mnist1d

```

```python

import numpy as np
import os
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import mnist1d
import random
from IPython.display import display, clear_output

```

```python

args = mnist1d.data.get_dataset_args()
data = mnist1d.data.get_dataset(args, path='./mnist1d_data.pkl', download=False, regenerate=False)

# The training and test input and outputs are in
# data['x'], data['y'], data['x_test'], and data['y_test']
print("Examples in training set: {}".format(len(data['y'])))
print("Examples in test set: {}".format(len(data['y_test'])))
print("Length of each example: {}".format(data['x'].shape[-1]))

```

Define the network

```python

D_i = 40    # Input dimensions
D_k = 300   # Hidden dimensions
D_o = 10    # Output dimensions

model = nn.Sequential(
nn.Linear(D_i, D_k),
nn.ReLU(),
nn.Linear(D_k, D_k),
nn.ReLU(),
nn.Linear(D_k, D_k),
nn.ReLU(),
nn.Linear(D_k, D_k),
nn.ReLU(),
nn.Linear(D_k, D_o))

```

```python

# He initialization of weights
def weights_init(layer_in):
  if isinstance(layer_in, nn.Linear):
    nn.init.kaiming_uniform_(layer_in.weight)
    layer_in.bias.data.fill_(0.0)

```

```python

def train_model(train_data_x, train_data_y, n_epoch):
  # choose cross entropy loss function (equation 5.24 in the loss notes)
  loss_function = nn.CrossEntropyLoss()
  # construct SGD optimizer and initialize learning rate and momentum
  optimizer = torch.optim.SGD(model.parameters(), lr = 0.02, momentum=0.9)
  # object that decreases learning rate by half every 20 epochs
  scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
  # create 100 dummy data points and store in data loader class
  x_train = torch.tensor(train_data_x.transpose().astype('float32'))
  y_train = torch.tensor(train_data_y.astype('long'))

  # load the data into a class that creates the batches
  data_loader = DataLoader(TensorDataset(x_train,y_train), batch_size=100, shuffle=True, worker_init_fn=np.random.seed(1))

  # Initialize model weights
  model.apply(weights_init)

  # store the loss and the % correct at each epoch
  losses_train = np.zeros((n_epoch))

  for epoch in range(n_epoch):
    # loop over batches
    for i, data in enumerate(data_loader):
      # retrieve inputs and labels for this batch
      x_batch, y_batch = data
      # zero the parameter gradients
      optimizer.zero_grad()
      # forward pass -- calculate model output
      pred = model(x_batch)
      # compute the loss
      loss = loss_function(pred, y_batch)
      # backward pass
      loss.backward()
      # SGD update
      optimizer.step()

    # Run whole dataset to get statistics -- normally wouldn't do this
    pred_train = model(x_train)
    _, predicted_train_class = torch.max(pred_train.data, 1)
    losses_train[epoch] = loss_function(pred_train, y_train).item()
    if epoch % 5 == 0:
        clear_output(wait=True)
        display("Epoch %d, train loss %3.3f"%(epoch, losses_train[epoch]))

    # tell scheduler to consider updating learning rate
    scheduler.step()

  return losses_train

```

```python

# Load in the data
train_data_x = data['x'].transpose()
train_data_y = data['y']
# Print out sizes
print("Train data: %d examples (columns), each of which has %d dimensions (rows)"%((train_data_x.shape[1],train_data_x.shape[0])))

```

```python

# Compute loss for proper data  and plot
n_epoch = 60
loss_true_labels = train_model(train_data_x, train_data_y, n_epoch)
# Plot the results
fig, ax = plt.subplots()
ax.plot(loss_true_labels,'r-',label='true_labels')
# ax.set_ylim(0,0.7); ax.set_xlim(0,n_epoch)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.legend()
plt.show()

```

```python

# TODO -- Randomize the input data (train_data_x), but retain overall mean and variance
# Replace this line
train_data_x_randomized = np.copy(train_data_x)

```

```python

# Compute loss for true labels and plot
n_epoch = 60
loss_randomized_data = train_model(train_data_x_randomized, train_data_y, n_epoch)
# Plot the results
fig, ax = plt.subplots()
ax.plot(loss_true_labels,'r-',label='true_labels')
ax.plot(loss_randomized_data,'b-',label='random_data')
# ax.set_ylim(0,0.7); ax.set_xlim(0,n_epoch)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.legend()
plt.show()

```

```python

# TODO -- Permute the labels
# Replace this line:
train_data_y_permuted = np.copy(train_data_y)

```

```python

# Compute loss for true labels and plot
n_epoch = 60
loss_permuted_labels = train_model(train_data_x, train_data_y_permuted, n_epoch)
# Plot the results
fig, ax = plt.subplots()
ax.plot(loss_true_labels,'r-',label='true_labels')
ax.plot(loss_randomized_data,'b-',label='random_data')
ax.plot(loss_permuted_labels,'g-',label='random_labels')
# ax.set_ylim(0,0.7); ax.set_xlim(0,n_epoch)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.legend()
plt.show()

```

***


<a href="https://colab.research.google.com/github/udlbook/udlbook/blob/main/Notebooks/Chap20/20_2_Full_Batch_Gradient_Descent.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# **Notebook 20.2: Full Batch Gradient Descent**

This notebook investigates training a network with full batch gradient descent as in figure 20.2.  There is also a version (notebook takes a long time to run), but this didn't speed it up much for me.  If you run out of CoLab time,  you'll need to download the Python file and run locally.

Work through the cells below, running each cell in turn. In various places you will see the words "TODO". Follow the instructions at these places and make predictions about what is going to happen or write code to complete the functions.

Contact me at udlbookmail@gmail.com if you find any mistakes or have any suggestions.

```python

# Run this if you're in a Colab to install MNIST 1D repository
!pip install git+https://github.com/greydanus/mnist1d

```

```python

import numpy as np
import os
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import mnist1d
import random
from IPython.display import display, clear_output

```

```python

args = mnist1d.data.get_dataset_args()
data = mnist1d.data.get_dataset(args, path='./mnist1d_data.pkl', download=False, regenerate=False)

# The training and test input and outputs are in
# data['x'], data['y']
print("Examples in training set: {}".format(len(data['y'])))
print("Length of each example: {}".format(data['x'].shape[-1]))

```

Define the network

```python

# Data is length forty, and there are 10 classes
D_i = 40
D_o = 10

# create model with one hidden layer and 298 hidden units
model_1_layer = nn.Sequential(
nn.Linear(D_i, 298),
nn.ReLU(),
nn.Linear(298, D_o))


# TODO -- create model with three hidden layers and 100 hidden units per layer
# Replace this line
model_2_layer = nn.Sequential(nn.Linear(D_i, D_o))



# TODO -- Create model with three hidden layers and 75 hidden units per layer
# Replace this line
model_3_layer = nn.Sequential(nn.Linear(D_i, D_o))



# TODO create model with four hidden layers and 63 hidden units per layer
# Replace this line
model_4_layer = nn.Sequential(nn.Linear(D_i, D_o))

```

```python

# He initialization of weights
def weights_init(layer_in):
  if isinstance(layer_in, nn.Linear):
    nn.init.kaiming_uniform_(layer_in.weight)
    layer_in.bias.data.fill_(0.0)

```

```python

def train_model(model, train_data_x, train_data_y, n_epoch):
  print("This is going to take a long time!")
  # choose cross entropy loss function (equation 5.24 in the loss notes)
  loss_function = nn.CrossEntropyLoss()
  # construct SGD optimizer and initialize learning rate to small value and momentum to 0
  optimizer = torch.optim.SGD(model.parameters(), lr = 0.0025, momentum=0.0)
  # create 100 dummy data points and store in data loader class
  x_train = torch.tensor(train_data_x.transpose().astype('float32'))
  y_train = torch.tensor(train_data_y.astype('long'))

  # load the data into a class that creates the batches -- full batch as there are 4000 examples
  data_loader = DataLoader(TensorDataset(x_train,y_train), batch_size=4000, shuffle=False, worker_init_fn=np.random.seed(1))

  # Initialize model weights
  model.apply(weights_init)

  # store the errors percentage at each point
  errors_train = np.zeros((n_epoch))

  for epoch in range(n_epoch):
    # loop over batches
    for i, data in enumerate(data_loader):
      # retrieve inputs and labels for this batch
      x_batch, y_batch = data
      # zero the parameter gradients
      optimizer.zero_grad()
      # forward pass -- calculate model output
      pred = model(x_batch)
      # compute the loss
      loss = loss_function(pred, y_batch)
      # Store the errors
      _, predicted_train_class = torch.max(pred.data, 1)
      errors_train[epoch] = 100 - 100 * (predicted_train_class == y_train).float().sum() / len(y_train)
      # backward pass
      loss.backward()
      # SGD update
      optimizer.step()

      if epoch % 10 == 0:
        clear_output(wait=True)
        display("Epoch %d, errors_train %3.3f"%(epoch, errors_train[epoch]))

  return errors_train

```

```python

# Load in the data
train_data_x = data['x'].transpose()
train_data_y = data['y']
# Print out sizes
print("Train data: %d examples (columns), each of which has %d dimensions (rows)"%((train_data_x.shape[1],train_data_x.shape[0])))

```

```python

# Train the models
errors_four_layers = train_model(model_4_layer, train_data_x, train_data_y, n_epoch=200000)

```

```python

errors_three_layers = train_model(model_3_layer, train_data_x, train_data_y, n_epoch=200000)

```

```python

errors_two_layers = train_model(model_2_layer, train_data_x, train_data_y, n_epoch=200000)

```

```python

errors_one_layer = train_model(model_1_layer, train_data_x, train_data_y, n_epoch=500000)

```

```python

# Plot the results
fig, ax = plt.subplots()
ax.plot(errors_one_layer,'r-',label='one layer')
ax.plot(errors_two_layers,'g-',label='two layers')
ax.plot(errors_three_layers,'b-',label='three layers')
ax.plot(errors_four_layers,'m-',label='four layers')
ax.set_ylim(0,100)
ax.set_xlabel('Epoch'); ax.set_ylabel('Percent error')
ax.legend()
plt.show()

```

***


<a href="https://colab.research.google.com/github/udlbook/udlbook/blob/main/Notebooks/Chap20/20_3_Lottery_Tickets.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# **Notebook 20.3: Lottery tickets**

This notebook investigates the phenomenon of lottery tickets as discussed in section 20.2.7.  This notebook is highly derivative of the MNIST-1D code hosted by Sam Greydanus at https://github.com/greydanus/mnist1d.   

Work through the cells below, running each cell in turn. In various places you will see the words "TODO". Follow the instructions at these places and make predictions about what is going to happen or write code to complete the functions.

Contact me at udlbookmail@gmail.com if you find any mistakes or have any suggestions.

```python

# Run this if you're in a Colab to install MNIST 1D repository
!pip install git+https://github.com/greydanus/mnist1d
!git clone https://github.com/greydanus/mnist1d

```

# Lottery tickets

Lottery tickets were first identified by [Frankle and Carbin (2018)](https://arxiv.org/abs/1803.03635).  They noted that after training a network, they could set the smaller weights to zero and clamp them there and retrain to get a network that was sparser (had fewer parameters) but could actually perform better.  So within the neural network there lie smaller sub-networks which are superior.  If we knew what these were, we could train them from scratch, but there is currently no way of finding out.

```python

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import mnist1d
import copy

```

## Get the MNIST1D dataset

```python

from mnist1d.data import get_dataset, get_dataset_args
from mnist1d.utils import set_seed, to_pickle, from_pickle

import sys ; sys.path.append('./mnist1d/notebooks')
from train import get_model_args, train_model

args = mnist1d.get_dataset_args()
data = mnist1d.get_dataset(args=args)  # by default, this will download a pre-made dataset from the GitHub repo

print("Examples in training set: {}".format(len(data['y'])))
print("Examples in test set: {}".format(len(data['y_test'])))
print("Length of each input: {}".format(data['x'].shape[-1]))
print("Number of classes: {}".format(len(data['templates']['y'])))

```

## Make an MLP that can be masked
These parameter-wise binary masks are how we will represent sparsity in this project. There's not a great PyTorch API for this yet, so here's a temporary solution.

```python

# Class to represent linear layer where some of the weights are forced to zero.
class SparseLinear(torch.nn.Module):
  def __init__(self, x_size, y_size):
    super(SparseLinear, self).__init__()
    self.linear = torch.nn.Linear(x_size, y_size)
    param_vec = torch.cat([p.flatten() for p in self.parameters()])
    self.mask = torch.ones_like(param_vec)

  def forward(self, x, apply_mask=True):
    if apply_mask:
      self.apply_mask()
    return self.linear(x)

  def update_mask(self, new_mask):
    self.mask = new_mask
    self.apply_mask()

  def apply_mask(self):
    self.vec2param(self.param2vec())

  def param2vec(self):
    vec = torch.cat([p.flatten() for p in self.parameters()])
    return self.mask * vec

  def vec2param(self, vec):
    pointer = 0
    for param in self.parameters():
      param_len = np.cumprod(param.shape)[-1]
      new_param = vec[pointer:pointer+param_len].reshape(param.shape)
      param.data = new_param.data
      pointer += param_len

# A two layer residual network where the linear layers are sparse
class SparseMLP(torch.nn.Module):
  def __init__(self, input_size, output_size, hidden_size=100):
    super(SparseMLP, self).__init__()
    self.linear1 = SparseLinear(input_size, hidden_size)
    self.linear2 = SparseLinear(hidden_size, hidden_size)
    self.linear3 = SparseLinear(hidden_size, output_size)
    self.layers = [self.linear1, self.linear2, self.linear3]

  def forward(self, x):
    h = torch.relu(self.linear1(x))
    h = h + torch.relu(self.linear2(h))
    h = self.linear3(h)
    return h

  def get_layer_masks(self):
    return [l.mask for l in self.layers]

  def set_layer_masks(self, new_masks):
    for i, l in enumerate(self.layers):
      l.update_mask(new_masks[i])

  def get_layer_vecs(self):
    return [l.param2vec() for l in self.layers]

  def set_layer_vecs(self, vecs):
    for i, l in enumerate(self.layers):
      l.vec2param(vecs[i])

```

Now we need a routine that takes the weights from the model and returns a mask that identifies the positions with the lowest magnitude.  These will be the weights that we mask.

```python

# absolute weights -- absolute values of all the weights from the model in a long vector
# percent_sparse: how much to sparsify the model
def get_mask(absolute_weights, percent_sparse):
  # TODO -- Write a function that returns a mask that has a zero
  # everywhere for the lowest "percent_sparse" of the absolute weights.
  # E.g. if absolute_weights contains [5,6,0,1,7] and we want percent_sparse of 40%,
  # we would return [1,1,0,0,1]
  # Remember that these are torch tensors and not numpy arrays
  # Replace this function:
  mask = torch.ones_like(absolute_weights)


  return mask

```

## The prune-and-retrain cycle
This is the core method for finding a lottery ticket. We train a model for a fixed number of epochs, prune it, and then re-train and re-prune. We repeat this cycle until we achieve the desired level of sparsity.

```python

def find_lottery_ticket(model, dataset, args, sparsity_schedule, criteria_fn=None, **kwargs):

  criteria_fn = lambda init_params, final_params: final_params.abs()
  init_params = model.get_layer_vecs()
  stats = {'train_losses':[], 'test_losses':[], 'train_accs':[], 'test_accs':[]}
  models = []
  for i, percent_sparse in enumerate(sparsity_schedule):

    # layer-wise pruning, where pruning heuristic is determined by criteria_fn
    final_params = model.get_layer_vecs()
    scores = [criteria_fn(ip, fp) for ip, fp in zip(init_params, final_params)]
    masks = [get_mask(s, percent_sparse) for s in scores]

    # update model with mask and init parameters
    model.set_layer_vecs(init_params)
    model.set_layer_masks(masks)

    # training process
    results = train_model(dataset, model, args)
    model = results['checkpoints'][-1]

    # store stats
    stats['train_losses'].append(results['train_losses'])
    stats['test_losses'].append(results['test_losses'])
    stats['train_accs'].append(results['train_acc'])
    stats['test_accs'].append(results['test_acc'])

    # print progress
    if (i+1) % 1 == 0:
      print('\tretrain #{}, sparsity {:.2f}, final_train_loss {:.3e}, max_acc {:.1f}, last_acc {:.1f}, mean_acc {:.1f}'
            .format(i+1, percent_sparse, results['train_losses'][-1], np.max(results['test_acc']),
            results['test_acc'][-1], np.mean(results['test_acc']) ))
      models.append(copy.deepcopy(model))

  stats = {k: np.stack(v) for k, v in stats.items()}
  return models, stats

```

## Choose hyperparameters

```python

# train settings
from train import get_model_args, train_model
model_args = get_model_args()
model_args.total_steps = 1501
model_args.hidden_size = 500
model_args.print_every = 5000 # print never
model_args.eval_every = 100
model_args.learning_rate = 2e-2
model_args.device = str('cpu')

```

Find the lottery ticket by repeatedly training and then pruning weights based on their magnitudes. We'll remove 1% of the weights each time. This is going to take half an hour or so.  Go and have lunch or whatever.

```python

# sparsity settings - we will train 100 models with progressively increasing sparsity
num_retrains = 100
sparsity_schedule = np.linspace(0,1.,num_retrains)

print("Magnitude pruning")
mnist1d.set_seed(model_args.seed)
model = SparseMLP(model_args.input_size, model_args.output_size, hidden_size=model_args.hidden_size)

criteria_fn = lambda init_params, final_params: final_params.abs()
lott_models, lott_stats = find_lottery_ticket(model, data, model_args, sparsity_schedule, criteria_fn=criteria_fn, prune_print_every=1)

```

```python

test_losses = lott_stats['test_losses'][:,-1]
test_accs = lott_stats['test_accs'][:,-1]

fig,ax = plt.subplots()
ax.plot(sparsity_schedule, test_losses,'r-')
ax.plot((sparsity_schedule[0], sparsity_schedule[-1]),(test_losses[0], test_losses[0]),'k--',label='dense')
ax.set_xlabel('Sparsity')
ax.set_ylabel('Loss')
ax.set_xlim(0,1)
ax.legend()
plt.show()

fig,ax = plt.subplots()
ax.plot(sparsity_schedule, 100-test_accs,'r-')
ax.plot((sparsity_schedule[0], sparsity_schedule[-1]),(100-test_accs[0], 100-test_accs[0]),'k--',label='dense')
ax.set_xlabel('Sparsity')
ax.set_ylabel('Percent Error')
ax.set_xlim(0,1)
ax.set_ylim(0,100)
ax.legend()
plt.show()

```

You should see that the test loss decreases and the errors decrease with more as the network gets sparser.  The dashed line represents the original dense (unpruned) network.  We have identified a simpler network that was "inside" the original network for which the results are superior.  Of course if we make it too sparse, then it gets worse again.

This phenomenon is explored much further in the original notebook by Sam Greydanus which can be found [here](https://github.com/greydanus/mnist1d).

***


<a href="https://colab.research.google.com/github/udlbook/udlbook/blob/main/Notebooks/Chap20/20_4_Adversarial_Attacks.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# **Notebook 20.4: Adversarial attacks**

This notebook builds uses the network for classification of MNIST from Notebook 10.5.  The code is adapted from https://nextjournal.com/gkoehler/pytorch-mnist, and uses the fast gradient sign attack of [Goodfellow et al. (2015)](https://arxiv.org/abs/1412.6572).  Having trained, the network, we search for adversarial examples -- inputs which look very similar to class A, but are mistakenly classified as class B.  We do this by starting with a correctly classified example and perturbing it according to the gradients of the network so that the output changes.

Work through the cells below, running each cell in turn. In various places you will see the words "TODO". Follow the instructions at these places and make predictions about what is going to happen or write code to complete the functions.

Contact me at udlbookmail@gmail.com if you find any mistakes or have any suggestions.

```python

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random

```

```python

# Run this once to load the train and test data straight into a dataloader class
# that will provide the batches
batch_size_train = 64
batch_size_test = 1000
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

```

```python

# Let's draw some of the training data
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
plt.show()

```

Define the network.  This is a more typical way to define a network than the sequential structure.  We define a class for the network, and define the parameters in the constructor.  Then we use a function called forward to actually run the network.  It's easy to see how you might use residual connections in this format.

```python

from os import X_OK

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x)
        return x

```

```python

# He initialization of weights
def weights_init(layer_in):
  if isinstance(layer_in, nn.Linear):
    nn.init.kaiming_uniform_(layer_in.weight)
    layer_in.bias.data.fill_(0.0)

```

```python

# Create network
model = Net()
# Initialize model weights
model.apply(weights_init)
# Define optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

```

```python

# Main training routine
def train(epoch):
  model.train()
  # Get each
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    # Store results
    if batch_idx % 10 == 0:
      print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()))

```

```python

# Run on test data
def test():
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = model(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

```

```python

# Get initial performance
test()
# Train for three epochs
n_epochs = 3
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()

```

```python

# Run network on data we got before and show predictions
output = model(example_data)

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Prediction: {}".format(
    output.data.max(1, keepdim=True)[1][i].item()))
  plt.xticks([])
  plt.yticks([])
plt.show()

```

This is the code that does the adversarial attack. It is adapted from [here](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html).   It is an example of the fast gradient sign method (FGSM), which modifies the data by



*   Calculating the derivative $\partial L/\partial \mathbf{x}$ of the loss $L$ with respect to the input data $\mathbf{x}$.
*   Finds the sign of the gradient at each point (making a tensor the same size as $\mathbf{x}$ with a one where it was positive and minus one where it was negative.  
*   Multiplying this vector by $\epsilon$ and adding it back to the original data

```python

# FGSM attack code.
def fgsm_attack(x, epsilon, dLdx):
    # TODO -- write this function
    # Get the sign of the gradient
    # Add epsilon times the size of gradient to x
    # Replace this line
    x_modified = torch.zeros_like(x)

    # Return the perturbed image
    return x_modified

```

```python

no_examples = 3
epsilon = 0.5
for i in range(no_examples):
  # Reset gradients
  optimizer.zero_grad()

  # Get the i'th data example
  x = example_data[i,:,:,:]
  # Add an extra dimension back to the beginning
  x= x[None, :,:,:]
  x.requires_grad = True
  # Get the i'th target
  y = torch.ones(1, dtype=torch.long) * example_targets[i]

  # Run the model
  output = model(x)
  # Compute the loss
  loss = F.nll_loss(output, y)
  # Back propagate
  loss.backward()

  # Collect ``datagrad``
  dLdx = x.grad.data

  # Call FGSM Attack
  x_prime = fgsm_attack(x, epsilon, dLdx)

  # Re-classify the perturbed image
  output_prime = model(x_prime)

  x = x.detach().numpy()
  fig = plt.figure()
  plt.subplot(1,2,1)
  plt.tight_layout()
  plt.imshow(x[0][0], cmap='gray', interpolation='none')
  plt.title("Original Prediction: {}".format(
    output.data.max(1, keepdim=True)[1][0].item()))
  plt.xticks([])
  plt.yticks([])

  plt.subplot(1,2,2)
  plt.tight_layout()
  plt.imshow(x_prime[0][0].detach().numpy(), cmap='gray', interpolation='none')
  plt.title("Perturbed Prediction: {}".format(
    output_prime.data.max(1, keepdim=True)[1][0].item()))
  plt.xticks([])
  plt.yticks([])

plt.show()

```

Although we have only added a small amount of noise, the model is fooled into thinking that these images come from a different class.

TODO -- Modify the attack so that it iteratively perturbs the data. i.e., so we take a small step epsilon, then re-calculate the gradient and take another small step according to the new gradient signs.

***
