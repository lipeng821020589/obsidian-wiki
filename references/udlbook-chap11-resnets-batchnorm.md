# Chap 11: 残差网络与批归一化 (ResNets & BatchNorm)

> UDLbook 精读笔记
>
> **官方资源**: [GitHub Notebooks](https://github.com/udlbook/udlbook/tree/main/Notebooks/Chap11)

---

## Notebook 列表

- **破碎梯度**: `Chap11/11_1_Shattered_Gradients.ipynb`
- **残差网络**: `Chap11/11_2_Residual_Networks.ipynb`
- **批归一化**: `Chap11/11_3_Batch_Normalization.ipynb`

---

## 内容

# **Notebook 11.1: Shattered gradients**

This notebook investigates the phenomenon of shattered gradients as discussed in section 11.1.1.  It replicates some of the experiments in [Balduzzi et al. (2017)](https://arxiv.org/abs/1702.08591).

Work through the cells below, running each cell in turn. In various places you will see the words "TODO". Follow the instructions at these places and make predictions about what is going to happen or write code to complete the functions.

Contact me at udlbookmail@gmail.com if you find any mistakes or have any suggestions.

```python

import numpy as np
import matplotlib.pyplot as plt

```

First let's define a neural network. We'll initialize both the weights and biases randomly with Glorot initialization (He initialization without the factor of two)

```python

# K is depth, D is number of hidden units in each layer
def init_params(K, D):
  # Set seed so we always get the same random numbers
  np.random.seed(1)

  # Input layer
  D_i = 1
  # Output layer
  D_o = 1

  # Glorot initialization
  sigma_sq_omega = 1.0/D

  # Make empty lists
  all_weights = [None] * (K+1)
  all_biases = [None] * (K+1)

  # Create parameters for input and output layers
  all_weights[0] = np.random.normal(size=(D, D_i))*np.sqrt(sigma_sq_omega)
  all_weights[-1] = np.random.normal(size=(D_o, D)) * np.sqrt(sigma_sq_omega)
  all_biases[0] = np.random.normal(size=(D,1))* np.sqrt(sigma_sq_omega)
  all_biases[-1]= np.random.normal(size=(D_o,1))* np.sqrt(sigma_sq_omega)

  # Create intermediate layers
  for layer in range(1,K):
    all_weights[layer] = np.random.normal(size=(D,D))*np.sqrt(sigma_sq_omega)
    all_biases[layer] = np.random.normal(size=(D,1))* np.sqrt(sigma_sq_omega)

  return all_weights, all_biases

```

The next two functions define the forward pass of the algorithm

```python

# Define the Rectified Linear Unit (ReLU) function
def ReLU(preactivation):
  activation = preactivation.clip(0.0)
  return activation

def forward_pass(net_input, all_weights, all_biases):

  # Retrieve number of layers
  K = len(all_weights) -1

  # We'll store the pre-activations at each layer in a list "all_f"
  # and the activations in a second list[all_h].
  all_f = [None] * (K+1)
  all_h = [None] * (K+1)

  #For convenience, we'll set
  # all_h[0] to be the input, and all_f[K] will be the output
  all_h[0] = net_input

  # Run through the layers, calculating all_f[0...K-1] and all_h[1...K]
  for layer in range(K):
      # Update preactivations and activations at this layer according to eqn 7.5
      all_f[layer] = all_biases[layer] + np.matmul(all_weights[layer], all_h[layer])
      all_h[layer+1] = ReLU(all_f[layer])

  # Compute the output from the last hidden layer
  all_f[K] = all_biases[K] + np.matmul(all_weights[K], all_h[K])

  # Retrieve the output
  net_output = all_f[K]

  return net_output, all_f, all_h

```

The next two functions compute the gradient of the output with respect to the input using the back propagation algorithm.

```python

# We'll need the indicator function
def indicator_function(x):
  x_in = np.array(x)
  x_in[x_in>=0] = 1
  x_in[x_in<0] = 0
  return x_in

# Main backward pass routine
def calc_input_output_gradient(x_in, all_weights, all_biases):

  #Retrieve number of layers
  K = len(all_weights) -1

  # Run the forward pass
  y, all_f, all_h = forward_pass(x_in, all_weights, all_biases)

  # We'll store the derivatives dl_dweights and dl_dbiases in lists as well
  all_dl_dweights = [None] * (K+1)
  all_dl_dbiases = [None] * (K+1)
  # And we'll store the derivatives of the loss with respect to the activation and preactivations in lists
  all_dl_df = [None] * (K+1)
  all_dl_dh = [None] * (K+1)
  # Again for convenience we'll stick with the convention that all_h[0] is the net input and all_f[k] in the net output

  # Compute derivatives of net output with respect to loss
  all_dl_df[K] = np.ones_like(all_f[K])

  # Now work backwards through the network
  for layer in range(K,-1,-1):
    all_dl_dbiases[layer] = np.array(all_dl_df[layer])
    all_dl_dweights[layer] = np.matmul(all_dl_df[layer], all_h[layer].transpose())

    all_dl_dh[layer] = np.matmul(all_weights[layer].transpose(), all_dl_df[layer])

    if layer > 0:
      all_dl_df[layer-1] = indicator_function(all_f[layer-1]) * all_dl_dh[layer]


  return all_dl_dh[0],y

```

Double check we have the gradient correct using finite differences

```python

D = 200; K = 3
# Initialize parameters
all_weights, all_biases = init_params(K,D)

x = np.ones((1,1))
dydx,y = calc_input_output_gradient(x, all_weights, all_biases)

# Offset for finite gradients
delta = 0.00000001
x1 = x
y1,*_ = forward_pass(x1, all_weights, all_biases)
x2 = x+delta
y2,*_ = forward_pass(x2, all_weights, all_biases)
# Finite difference calculation
dydx_fd = (y2-y1)/delta

print("Gradient calculation=%f, Finite difference gradient=%f"%(dydx.squeeze(),dydx_fd.squeeze()))

```

Helper function that computes the derivatives for a 1D array of input values and plots them.

```python

def plot_derivatives(K, D):

  # Initialize parameters
  all_weights, all_biases = init_params(K,D)

  x_in = np.arange(-2,2, 4.0/256.0)
  x_in = np.resize(x_in, (1,len(x_in)))
  dydx,y = calc_input_output_gradient(x_in, all_weights, all_biases)

  fig,ax = plt.subplots()
  ax.plot(np.squeeze(x_in), np.squeeze(dydx), 'b-')
  ax.set_xlim(-2,2)
  ax.set_xlabel(r'Input, $x$')
  ax.set_ylabel(r'Gradient, $dy/dx$')
  ax.set_title('No layers = %d'%(K))
  plt.show()

```

```python

# Build a model with one hidden layer and 200 neurons and plot derivatives
D = 200; K = 1
plot_derivatives(K,D)

# TODO -- Interpret this result
# Why does the plot have some flat regions?

# TODO -- Add code to plot the derivatives for models with 24 and 50 hidden layers
# with 200 neurons per layer

# TODO -- Why does this graph not have visible flat regions?

# TODO -- Why does the magnitude of the gradients decrease as we increase the number
# of hidden layers

# TODO -- Do you find this a convincing replication of the experiment in the original paper? (I don't)
# Can you help me find why I have failed to replicate this result?  udlbookmail@gmail.com

```

Let's look at the autocorrelation function now

```python

def autocorr(dydx):
    # TODO -- compute the autocorrelation function
    # Use the numpy function "correlate" with the mode set to "same"
    # Replace this line:
    ac = np.ones((256,1))

    return ac

```

Helper function to plot the autocorrelation function and normalize so correlation is one with offset of zero

```python

def plot_autocorr(K, D):

  # Initialize parameters
  all_weights, all_biases = init_params(K,D)

  x_in = np.arange(-2.0,2.0, 4.0/256)
  x_in = np.resize(x_in, (1,len(x_in)))
  dydx,y = calc_input_output_gradient(x_in, all_weights, all_biases)
  ac = autocorr(np.squeeze(dydx))
  ac = ac / ac[128]

  y = ac[128:]
  x = np.squeeze(x_in)[128:]
  fig,ax = plt.subplots()
  ax.plot(x,y, 'b-')
  ax.set_xlim([0,2])
  ax.set_xlabel('Distance')
  ax.set_ylabel('Autocorrelation')
  ax.set_title('No layers = %d'%(K))
  plt.show()

```

```python

# Plot the autocorrelation functions
D = 200; K =1
plot_autocorr(K,D)
D = 200; K =50
plot_autocorr(K,D)

# TODO -- Do you find this a convincing replication of the experiment in the original paper? (I don't)
# Can you help me find why I have failed to replicate this result?

```

***


# **Notebook 11.2: Residual Networks**

This notebook adapts the networks for MNIST1D to use residual connections.

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

```python

# Load in the data
train_data_x = data['x'].transpose()
train_data_y = data['y']
val_data_x = data['x_test'].transpose()
val_data_y = data['y_test']
# Print out sizes
print("Train data: %d examples (columns), each of which has %d dimensions (rows)"%((train_data_x.shape[1],train_data_x.shape[0])))
print("Validation data: %d examples (columns), each of which has %d dimensions (rows)"%((val_data_x.shape[1],val_data_x.shape[0])))

```

Define the network

```python

# There are 40 input dimensions and 10 output dimensions for this data
# The inputs correspond to the 40 offsets in the MNIST1D template.
D_i = 40
# The outputs correspond to the 10 digits
D_o = 10


# We will adapt this model to have residual connections around the linear layers
# This is the same model we used in practical 8.1, but we can't use the sequential
# class for residual networks (which aren't strictly sequential).  Hence, I've rewritten
# it as a model that inherits from a base class

class ResidualNetwork(torch.nn.Module):
  def __init__(self, input_size, output_size, hidden_size=100):
    super(ResidualNetwork, self).__init__()
    self.linear1 = nn.Linear(input_size, hidden_size)
    self.linear2 = nn.Linear(hidden_size, hidden_size)
    self.linear3 = nn.Linear(hidden_size, hidden_size)
    self.linear4 = nn.Linear(hidden_size, output_size)
    print("Initialized MLPBase model with {} parameters".format(self.count_params()))

  def count_params(self):
    return sum([p.view(-1).shape[0] for p in self.parameters()])

# TODO -- Add residual connections to this model
# The order of operations within each block should similar to figure 11.5b
# ie., linear1 first, ReLU+linear2 in first residual block, ReLU+linear3 in second residual block), linear4 at end
# Replace this function
  def forward(self, x):
    h1 = self.linear1(x).relu()
    h2 = self.linear2(h1).relu()
    h3 = self.linear3(h2).relu()
    return self.linear4(h3)

```

```python

# He initialization of weights
def weights_init(layer_in):
  if isinstance(layer_in, nn.Linear):
    nn.init.kaiming_uniform_(layer_in.weight)
    layer_in.bias.data.fill_(0.0)

```

```python

#Define the model
model = ResidualNetwork(40, 10)

# choose cross entropy loss function (equation 5.24 in the loss notes)
loss_function = nn.CrossEntropyLoss()
# construct SGD optimizer and initialize learning rate and momentum
optimizer = torch.optim.SGD(model.parameters(), lr = 0.05, momentum=0.9)
# object that decreases learning rate by half every 20 epochs
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
# convert data to torch tensors
x_train = torch.tensor(train_data_x.transpose().astype('float32'))
y_train = torch.tensor(train_data_y.astype('long'))
x_val= torch.tensor(val_data_x.transpose().astype('float32'))
y_val = torch.tensor(val_data_y.astype('long'))

# load the data into a class that creates the batches
data_loader = DataLoader(TensorDataset(x_train,y_train), batch_size=100, shuffle=True, worker_init_fn=np.random.seed(1))

# Initialize model weights
model.apply(weights_init)

# loop over the dataset n_epoch times
n_epoch = 100
# store the loss and the % correct at each epoch
losses_train = np.zeros((n_epoch))
errors_train = np.zeros((n_epoch))
losses_val = np.zeros((n_epoch))
errors_val = np.zeros((n_epoch))

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
  pred_val = model(x_val)
  _, predicted_train_class = torch.max(pred_train.data, 1)
  _, predicted_val_class = torch.max(pred_val.data, 1)
  errors_train[epoch] = 100 - 100 * (predicted_train_class == y_train).float().sum() / len(y_train)
  errors_val[epoch]= 100 - 100 * (predicted_val_class == y_val).float().sum() / len(y_val)
  losses_train[epoch] = loss_function(pred_train, y_train).item()
  losses_val[epoch]= loss_function(pred_val, y_val).item()
  print(f'Epoch {epoch:5d}, train loss {losses_train[epoch]:.6f}, train error {errors_train[epoch]:3.2f},  val loss {losses_val[epoch]:.6f}, percent error {errors_val[epoch]:3.2f}')

  # tell scheduler to consider updating learning rate
  scheduler.step()

```

```python

# Plot the results
fig, ax = plt.subplots()
ax.plot(errors_train,'r-',label='train')
ax.plot(errors_val,'b-',label='test')
ax.set_ylim(0,100); ax.set_xlim(0,n_epoch)
ax.set_xlabel('Epoch'); ax.set_ylabel('Error')
ax.set_title('TrainError %3.2f, Val Error %3.2f'%(errors_train[-1],errors_val[-1]))
ax.legend()
plt.show()

```

The primary motivation of residual networks is to allow training of much deeper networks.   

TODO: Try running this network with and without the residual connections.  Does adding the residual connections change the performance?

***


# **Notebook 11.3: Batch normalization**

This notebook investigates the use of batch normalization in residual networks.

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

```python

# Load in the data
train_data_x = data['x'].transpose()
train_data_y = data['y']
val_data_x = data['x_test'].transpose()
val_data_y = data['y_test']
# Print out sizes
print("Train data: %d examples (columns), each of which has %d dimensions (rows)"%((train_data_x.shape[1],train_data_x.shape[0])))
print("Validation data: %d examples (columns), each of which has %d dimensions (rows)"%((val_data_x.shape[1],val_data_x.shape[0])))

```

```python

def print_variance(name, data):
  # First dimension(rows) is batch elements
  # Second dimension(columns) is neurons.
  np_data = data.detach().numpy()
  # Compute variance across neurons and average these variances over members of the batch
  neuron_variance = np.mean(np.var(np_data, axis=0))
  # Print out the name and the variance
  print("%s variance=%f"%(name,neuron_variance))

```

```python

# He initialization of weights
def weights_init(layer_in):
  if isinstance(layer_in, nn.Linear):
    nn.init.kaiming_uniform_(layer_in.weight)
    layer_in.bias.data.fill_(0.0)

```

```python

def run_one_step_of_model(model, x_train, y_train):
  # choose cross entropy loss function (equation 5.24 in the loss notes)
  loss_function = nn.CrossEntropyLoss()
  # construct SGD optimizer and initialize learning rate and momentum
  optimizer = torch.optim.SGD(model.parameters(), lr = 0.05, momentum=0.9)

  # load the data into a class that creates the batches
  data_loader = DataLoader(TensorDataset(x_train,y_train), batch_size=200, shuffle=True, worker_init_fn=np.random.seed(1))

  # Initialize model weights
  model.apply(weights_init)

  # Get a batch
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
    # Break out of this loop -- we just want to see the first
    # iteration, but usually we would continue
    break

```

```python

# convert training data to torch tensors
x_train = torch.tensor(train_data_x.transpose().astype('float32'))
y_train = torch.tensor(train_data_y.astype('long'))

```

```python

# This is a simple residual model with 5 residual branches in a row
class ResidualNetwork(torch.nn.Module):
  def __init__(self, input_size, output_size, hidden_size=100):
    super(ResidualNetwork, self).__init__()
    self.linear1 = nn.Linear(input_size, hidden_size)
    self.linear2 = nn.Linear(hidden_size, hidden_size)
    self.linear3 = nn.Linear(hidden_size, hidden_size)
    self.linear4 = nn.Linear(hidden_size, hidden_size)
    self.linear5 = nn.Linear(hidden_size, hidden_size)
    self.linear6 = nn.Linear(hidden_size, hidden_size)
    self.linear7 = nn.Linear(hidden_size, output_size)

  def count_params(self):
    return sum([p.view(-1).shape[0] for p in self.parameters()])

  def forward(self, x):
    print_variance("Input",x)
    f = self.linear1(x)
    print_variance("First preactivation",f)
    res1 = f+ self.linear2(f.relu())
    print_variance("After first residual connection",res1)
    res2 = res1 + self.linear3(res1.relu())
    print_variance("After second residual connection",res2)
    res3 = res2 + self.linear4(res2.relu())
    print_variance("After third residual connection",res3)
    res4 = res3 + self.linear5(res3.relu())
    print_variance("After fourth residual connection",res4)
    res5 = res4 + self.linear6(res4.relu())
    print_variance("After fifth residual connection",res5)
    return self.linear7(res5)

```

```python

# Define the model and run for one step
# Monitoring the variance at each point in the network
n_hidden = 100
n_input = 40
n_output = 10
model = ResidualNetwork(n_input, n_output, n_hidden)
run_one_step_of_model(model, x_train, y_train)

```

Notice that the variance roughly doubles at each step so it increases exponentially as in figure 11.6b in the book.

```python

# TODO Adapt the residual network below to add a batch norm operation
# before the contents of each residual link as in figure 11.6c in the book
# Use the torch function nn.BatchNorm1d
class ResidualNetworkWithBatchNorm(torch.nn.Module):
  def __init__(self, input_size, output_size, hidden_size=100):
    super(ResidualNetworkWithBatchNorm, self).__init__()
    self.linear1 = nn.Linear(input_size, hidden_size)
    self.linear2 = nn.Linear(hidden_size, hidden_size)
    self.linear3 = nn.Linear(hidden_size, hidden_size)
    self.linear4 = nn.Linear(hidden_size, hidden_size)
    self.linear5 = nn.Linear(hidden_size, hidden_size)
    self.linear6 = nn.Linear(hidden_size, hidden_size)
    self.linear7 = nn.Linear(hidden_size, output_size)

  def count_params(self):
    return sum([p.view(-1).shape[0] for p in self.parameters()])

  def forward(self, x):
    print_variance("Input",x)
    f = self.linear1(x)
    print_variance("First preactivation",f)
    res1 = f+ self.linear2(f.relu())
    print_variance("After first residual connection",res1)
    res2 = res1 + self.linear3(res1.relu())
    print_variance("After second residual connection",res2)
    res3 = res2 + self.linear4(res2.relu())
    print_variance("After third residual connection",res3)
    res4 = res3 + self.linear5(res3.relu())
    print_variance("After fourth residual connection",res4)
    res5 = res4 + self.linear6(res4.relu())
    print_variance("After fifth residual connection",res5)
    return self.linear7(res5)

```

```python

# Define the model
n_hidden = 100
n_input = 40
n_output = 10
model = ResidualNetworkWithBatchNorm(n_input, n_output, n_hidden)
run_one_step_of_model(model, x_train, y_train)

```

Note that the variance now increases linearly as in figure 11.6c.

***
