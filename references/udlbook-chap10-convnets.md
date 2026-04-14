# Chap 10: 卷积神经网络 (ConvNets)

> UDLbook 精读笔记
>
> **官方资源**: [GitHub Notebooks](https://github.com/udlbook/udlbook/tree/main/Notebooks/Chap10)

---

## Notebook 列表

- **1D 卷积**: `Chap10/10_1_1D_Convolution.ipynb`
- **MNIST 1D 卷积**: `Chap10/10_2_Convolution_for_MNIST_1D.ipynb`
- **2D 卷积**: `Chap10/10_3_2D_Convolution.ipynb`
- **下采样与上采样**: `Chap10/10_4_Downsampling_and_Upsampling.ipynb`
- **MNIST 卷积**: `Chap10/10_5_Convolution_For_MNIST.ipynb`

---

## 内容

<a href="https://colab.research.google.com/github/udlbook/udlbook/blob/main/Notebooks/Chap10/10_1_1D_Convolution.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# **Notebook 10.1: 1D Convolution**

This notebook investigates 1D convolutional layers.

Work through the cells below, running each cell in turn. In various places you will see the words "TODO". Follow the instructions at these places and make predictions about what is going to happen or write code to complete the functions.

Contact me at udlbookmail@gmail.com if you find any mistakes or have any suggestions.

NOTE!!

If you have the first edition of the printed book, it mistakenly refers to a convolutional filter with no spaces between the elements (i.e. a normal filter without dilation) as having dilation zero.  Actually, the convention is (weirdly) that this has dilation one.  And when there is one space between the elements, this is dilation two.   This notebook reflects the correct convention and so will be out of sync with the printed book.  If this is confusing, check the [errata](https://github.com/udlbook/udlbook/blob/main/UDL_Errata.pdf) document.

```python

import numpy as np
import matplotlib.pyplot as plt

```

```python

# Define a signal that we can apply convolution to
x = [5.2, 5.3, 5.4, 5.1, 10.1, 10.3, 9.9, 10.3, 3.2, 3.4, 3.3, 3.1]

```

```python

# Draw the signal
fig,ax = plt.subplots()
ax.plot(x, 'k-')
ax.set_xlim(0,11)
ax.set_ylim(0, 12)
plt.show()

```

```python

# Now let's define a zero-padded convolution operation
# with a convolution kernel size of 3, a stride of 1, and a dilation of 1
# as in figure 10.2a-c.  Write it yourself, don't call a library routine!
# Don't forget that Python arrays are indexed from zero, not from 1 as in the book figures
def conv_3_1_1_zp(x_in, omega):
    x_out = np.zeros_like(x_in)
    # TODO -- write this function
    # replace this line
    x_out = x_out



    return x_out

```

Now let's see what kind of things convolution can do
First, it can average nearby values, smoothing the function:

```python

omega = [0.33,0.33,0.33]
h = conv_3_1_1_zp(x, omega)

# Check that you have computed this correctly
print(f"Sum of output is {np.sum(h):3.3}, should be 71.1")

# Draw the signal
fig,ax = plt.subplots()
ax.plot(x, 'k-',label='before')
ax.plot(h, 'r-',label='after')
ax.set_xlim(0,11)
ax.set_ylim(0, 12)
ax.legend()
plt.show()

```

Notice how the red function is a smoothed version of the black one as it has averaged adjacent values.  The first and last outputs are considerably lower than the original curve though.  Make sure that you understand why!<br><br>

With different weights, the convolution can be used to find sharp changes in the function:

```python

omega = [-0.5,0,0.5]
h2 = conv_3_1_1_zp(x, omega)

# Draw the signal
fig,ax = plt.subplots()
ax.plot(x, 'k-',label='before')
ax.plot(h2, 'r-',label='after')
ax.set_xlim(0,11)
# ax.set_ylim(0, 12)
ax.legend()
plt.show()

```

Notice that the convolution has a peak where the original function went up and trough where it went down.  It is roughly zero where the function is locally flat.  This convolution approximates a derivative. <br> <br>

Now let's define the convolutions from figure 10.3.

```python

# Now let's define a zero-padded convolution operation
# with a convolution kernel size of 3, a stride of 2, and a dilation of 1
# as in figure 10.3a-b.  Write it yourself, don't call a library routine!
def conv_3_2_1_zp(x_in, omega):
    x_out = np.zeros(int(np.ceil(len(x_in)/2)))
    # TODO -- write this function
    # replace this line
    x_out = x_out



    return x_out

```

```python

omega = [0.33,0.33,0.33]
h3 = conv_3_2_1_zp(x, omega)

# If you have done this right, the output length should be six and it should
# contain every other value from the original convolution with stride 1
print(h)
print(h3)

```

```python

# Now let's define a zero-padded convolution operation
# with a convolution kernel size of 5, a stride of 1, and a dilation of 1
# as in figure 10.3c.  Write it yourself, don't call a library routine!
def conv_5_1_1_zp(x_in, omega):
    x_out = np.zeros_like(x_in)
    # TODO -- write this function
    # replace this line
    x_out = x_out



    return x_out

```

```python

omega2 = [0.2, 0.2, 0.2, 0.2, 0.2]
h4 = conv_5_1_1_zp(x, omega2)

# Check that you have computed this correctly
print(f"Sum of output is {np.sum(h4):3.3}, should be 69.6")

# Draw the signal
fig,ax = plt.subplots()
ax.plot(x, 'k-',label='before')
ax.plot(h4, 'r-',label='after')
ax.set_xlim(0,11)
ax.set_ylim(0, 12)
ax.legend()
plt.show()

```

```python

# Finally let's define a zero-padded convolution operation
# with a convolution kernel size of 3, a stride of 1, and a dilation of 2
# as in figure 10.3d.  Write it yourself, don't call a library routine!
# Don't forget that Python arrays are indexed from zero, not from 1 as in the book figures
def conv_3_1_2_zp(x_in, omega):
    x_out = np.zeros_like(x_in)
    # TODO -- write this function
    # replace this line
    x_out = x_out


    return x_out

```

```python

omega = [0.33,0.33,0.33]
h5 = conv_3_1_2_zp(x, omega)

# Check that you have computed this correctly
print(f"Sum of output is {np.sum(h5):3.3}, should be 68.3")

# Draw the signal
fig,ax = plt.subplots()
ax.plot(x, 'k-',label='before')
ax.plot(h5, 'r-',label='after')
ax.set_xlim(0,11)
ax.set_ylim(0, 12)
ax.legend()
plt.show()

```

Finally, let's investigate representing convolutions as full matrices, and show we get the same answer.

```python

# Compute matrix in figure 10.4 d
def get_conv_mat_3_1_1_zp(n_out, omega):
  omega_mat = np.zeros((n_out,n_out))
  # TODO Fill in this matrix
  # Replace this line:
  omega_mat = omega_mat



  return omega_mat

```

```python

# Run original convolution
omega = np.array([-1.0,0.5,-0.2])
h6 = conv_3_1_1_zp(x, omega)
print(h6)

# If you have done this right, you should get the same answer
omega_mat = get_conv_mat_3_1_1_zp(len(x), omega)
h7 = np.matmul(omega_mat, x)
print(h7)

```

TODO:  What do you expect to happen if we apply the last convolution twice?  Can this be represented as a single convolution?  If so, then what is it?

***


<a href="https://colab.research.google.com/github/udlbook/udlbook/blob/main/Notebooks/Chap10/10_2_Convolution_for_MNIST_1D.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# **Notebook 10.2: Convolution for MNIST-1D**

This notebook investigates a 1D convolutional network for MNIST-1D as in figure 10.7 and 10.8a.

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


# TODO Create a model with the following layers
# 1. Convolutional layer, (input=length 40 and 1 channel, kernel size 3, stride 2, padding="valid", 15 output channels )
# 2. ReLU
# 3. Convolutional layer, (input=length 19 and 15 channels, kernel size 3, stride 2, padding="valid", 15 output channels )
# 4. ReLU
# 5. Convolutional layer, (input=length 9 and 15 channels, kernel size 3, stride 2, padding="valid", 15 output channels)
# 6. ReLU
# 7. Flatten (converts 4x15) to length 60
# 8. Linear layer (input size = 60, output size = 10)
# References:
# https://pytorch.org/docs/1.13/generated/torch.nn.Conv1d.html?highlight=conv1d#torch.nn.Conv1d
# https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html
# https://pytorch.org/docs/1.13/generated/torch.nn.Linear.html?highlight=linear#torch.nn.Linear

# NOTE THAT THE CONVOLUTIONAL LAYERS NEED TO TAKE THE NUMBER OF INPUT CHANNELS AS A PARAMETER
# AND NOT THE INPUT SIZE.

# Replace the following function:
model = nn.Sequential(
nn.Flatten(),
nn.Linear(40, 100),
nn.ReLU(),
nn.Linear(100, 100),
nn.ReLU(),
nn.Linear(100, 10))

```

```python

# He initialization of weights
def weights_init(layer_in):
  if isinstance(layer_in, nn.Linear):
    nn.init.kaiming_uniform_(layer_in.weight)
    layer_in.bias.data.fill_(0.0)

```

```python

# choose cross entropy loss function (equation 5.24 in the loss notes)
loss_function = nn.CrossEntropyLoss()
# construct SGD optimizer and initialize learning rate and momentum
optimizer = torch.optim.SGD(model.parameters(), lr = 0.05, momentum=0.9)
# object that decreases learning rate by half every 20 epochs
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
# create 100 dummy data points and store in data loader class
x_train = torch.tensor(train_data_x.transpose().astype('float32'))
y_train = torch.tensor(train_data_y.astype('long')).long()
x_val= torch.tensor(val_data_x.transpose().astype('float32'))
y_val = torch.tensor(val_data_y.astype('long')).long()

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
    pred = model(x_batch[:,None,:])
    # compute the loss
    loss = loss_function(pred, y_batch)
    # backward pass
    loss.backward()
    # SGD update
    optimizer.step()

  # Run whole dataset to get statistics -- normally wouldn't do this
  pred_train = model(x_train[:,None,:])
  pred_val = model(x_val[:,None,:])
  _, predicted_train_class = torch.max(pred_train.data, 1)
  _, predicted_val_class = torch.max(pred_val.data, 1)
  errors_train[epoch] = 100 - 100 * (predicted_train_class == y_train).float().sum() / len(y_train)
  errors_val[epoch]= 100 - 100 * (predicted_val_class == y_val).float().sum() / len(y_val)
  losses_train[epoch] = loss_function(pred_train, y_train).item()
  losses_val[epoch]= loss_function(pred_val, y_val).item()
  print(f'Epoch {epoch:5d}, train loss {losses_train[epoch]:.6f}, train error {errors_train[epoch]:3.2f},  val loss {losses_val[epoch]:.6f}, percent error {errors_val[epoch]:3.2f}')

  # tell scheduler to consider updating learning rate
  scheduler.step()

# Plot the results
fig, ax = plt.subplots()
ax.plot(errors_train,'r-',label='train')
ax.plot(errors_val,'b-',label='validation')
ax.set_ylim(0,100); ax.set_xlim(0,n_epoch)
ax.set_xlabel('Epoch'); ax.set_ylabel('Error')
ax.set_title('Part I: Validation Result %3.2f'%(errors_val[-1]))
ax.legend()
plt.show()

```

***


<a href="https://colab.research.google.com/github/udlbook/udlbook/blob/main/Notebooks/Chap10/10_3_2D_Convolution.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# **Notebook 10.3: 2D Convolution**

This notebook investigates the 2D convolution operation.  It asks you to hand code the convolution so we can be sure that we are computing the same thing as in PyTorch.  The next notebook uses the convolutional layers in PyTorch directly.

Work through the cells below, running each cell in turn. In various places you will see the words "TODO". Follow the instructions at these places and make predictions about what is going to happen or write code to complete the functions.

Contact me at udlbookmail@gmail.com if you find any mistakes or have any suggestions.

```python

import numpy as np
import torch
# Set to print in reasonable form
np.set_printoptions(precision=3, floatmode="fixed")
torch.set_printoptions(precision=3)

```

This routine performs convolution in PyTorch

```python

# Perform convolution in PyTorch
def conv_pytorch(image, conv_weights, stride=1, pad =1):
  # Convert image and kernel to tensors
  image_tensor = torch.from_numpy(image) # (batchSize, channelsIn, imageHeightIn, =imageWidthIn)
  conv_weights_tensor = torch.from_numpy(conv_weights) # (channelsOut, channelsIn, kernelHeight, kernelWidth)
  # Do the convolution
  output_tensor = torch.nn.functional.conv2d(image_tensor, conv_weights_tensor, stride=stride, padding=pad)
  # Convert back from PyTorch and return
  return(output_tensor.numpy()) # (batchSize channelsOut imageHeightOut imageHeightIn)

```

First we'll start with the simplest 2D convolution.  Just one channel in and one channel out.  A single image in the batch.

```python

# Perform convolution in numpy
def conv_numpy_1(image, weights, pad=1):

    # Perform zero padding
    if pad != 0:
        image = np.pad(image, ((0, 0), (0 ,0), (pad, pad), (pad, pad)),'constant')

    # Get sizes of image array and kernel weights
    batchSize,  channelsIn, imageHeightIn, imageWidthIn = image.shape
    channelsOut, channelsIn, kernelHeight, kernelWidth = weights.shape

    # Get size of output arrays
    imageHeightOut = np.floor(1 + imageHeightIn - kernelHeight).astype(int)
    imageWidthOut = np.floor(1 + imageWidthIn - kernelWidth).astype(int)

    # Create output
    out = np.zeros((batchSize, channelsOut, imageHeightOut, imageWidthOut), dtype=np.float32)

    # !!!!!! NOTE THERE IS A SUBTLETY HERE !!!!!!!!
    # I have padded the image with zeros above, so it is surrouned by a "ring" of zeros
    # That means that the image indexes are all off by one
    # This actually makes your code simpler

    for c_y in range(imageHeightOut):
      for c_x in range(imageWidthOut):
        for c_kernel_y in range(kernelHeight):
          for c_kernel_x in range(kernelWidth):
            # TODO -- Retrieve the image pixel and the weight from the convolution
            # Only one image in batch, one input channel and one output channel, so these indices should all be zero
            # Replace the two lines below
            this_pixel_value = 1.0
            this_weight = 1.0


            # Multiply these together and add to the output at this position
            out[0, 0, c_y, c_x] += np.sum(this_pixel_value * this_weight)

    return out

```

```python

# Set random seed so we always get same answer
np.random.seed(1)
n_batch = 1
image_height = 4
image_width = 6
channels_in = 1
kernel_size = 3
channels_out = 1

# Create random input image
input_image= np.random.normal(size=(n_batch, channels_in, image_height, image_width))
# Create random convolution kernel weights
conv_weights = np.random.normal(size=(channels_out, channels_in, kernel_size, kernel_size))

# Perform convolution using PyTorch
conv_results_pytorch = conv_pytorch(input_image, conv_weights, stride=1, pad=1)
print("PyTorch Results")
print(conv_results_pytorch)

# Perform convolution in numpy
print("Your results")
conv_results_numpy = conv_numpy_1(input_image, conv_weights)
print(conv_results_numpy)

```

Let's now add in the possibility of using different strides

```python

# Perform convolution in numpy
def conv_numpy_2(image, weights, stride=1, pad=1):

    # Perform zero padding
    if pad != 0:
        image = np.pad(image, ((0, 0), (0 ,0), (pad, pad), (pad, pad)),'constant')

    # Get sizes of image array and kernel weights
    batchSize,  channelsIn, imageHeightIn, imageWidthIn = image.shape
    channelsOut, channelsIn, kernelHeight, kernelWidth = weights.shape

    # Get size of output arrays
    imageHeightOut = np.floor(1 + (imageHeightIn - kernelHeight) / stride).astype(int)
    imageWidthOut = np.floor(1 + (imageWidthIn - kernelWidth) / stride).astype(int)

    # Create output
    out = np.zeros((batchSize, channelsOut, imageHeightOut, imageWidthOut), dtype=np.float32)

    for c_y in range(imageHeightOut):
      for c_x in range(imageWidthOut):
        for c_kernel_y in range(kernelHeight):
          for c_kernel_x in range(kernelWidth):
            # TODO -- Retrieve the image pixel and the weight from the convolution
            # Only one image in batch, one input channel and one output channel, so these indices should all be zero
            # Replace the two lines below
            this_pixel_value = 1.0
            this_weight = 1.0


            # Multiply these together and add to the output at this position
            out[0, 0, c_y, c_x] += np.sum(this_pixel_value * this_weight)

    return out

```

```python

# Set random seed so we always get same answer
np.random.seed(1)
n_batch = 1
image_height = 12
image_width = 10
channels_in = 1
kernel_size = 3
channels_out = 1
stride = 2

# Create random input image
input_image= np.random.normal(size=(n_batch, channels_in, image_height, image_width))
# Create random convolution kernel weights
conv_weights = np.random.normal(size=(channels_out, channels_in, kernel_size, kernel_size))

# Perform convolution using PyTorch
conv_results_pytorch = conv_pytorch(input_image, conv_weights, stride, pad=1)
print("PyTorch Results")
print(conv_results_pytorch)

# Perform convolution in numpy
print("Your results")
conv_results_numpy = conv_numpy_2(input_image, conv_weights, stride, pad=1)
print(conv_results_numpy)

```

Now we'll introduce multiple input and output channels

```python

# Perform convolution in numpy
def conv_numpy_3(image, weights, stride=1, pad=1):

    # Perform zero padding
    if pad != 0:
        image = np.pad(image, ((0, 0), (0 ,0), (pad, pad), (pad, pad)),'constant')

    # Get sizes of image array and kernel weights
    batchSize,  channelsIn, imageHeightIn, imageWidthIn = image.shape
    channelsOut, channelsIn, kernelHeight, kernelWidth = weights.shape

    # Get size of output arrays
    imageHeightOut = np.floor(1 + (imageHeightIn - kernelHeight) / stride).astype(int)
    imageWidthOut = np.floor(1 + (imageWidthIn - kernelWidth) / stride).astype(int)

    # Create output
    out = np.zeros((batchSize, channelsOut, imageHeightOut, imageWidthOut), dtype=np.float32)

    for c_y in range(imageHeightOut):
      for c_x in range(imageWidthOut):
        for c_channel_out in range(channelsOut):
          for c_channel_in in range(channelsIn):
            for c_kernel_y in range(kernelHeight):
              for c_kernel_x in range(kernelWidth):
                  # TODO -- Retrieve the image pixel and the weight from the convolution
                  # Only one image in batch so this index should be zero
                  # Replace the two lines below
                  this_pixel_value = 1.0
                  this_weight = 1.0

                  # Multiply these together and add to the output at this position
                  out[0, c_channel_out, c_y, c_x] += np.sum(this_pixel_value * this_weight)
    return out

```

```python

# Set random seed so we always get same answer
np.random.seed(1)
n_batch = 1
image_height = 4
image_width = 6
channels_in = 5
kernel_size = 3
channels_out = 2

# Create random input image
input_image= np.random.normal(size=(n_batch, channels_in, image_height, image_width))
# Create random convolution kernel weights
conv_weights = np.random.normal(size=(channels_out, channels_in, kernel_size, kernel_size))

# Perform convolution using PyTorch
conv_results_pytorch = conv_pytorch(input_image, conv_weights, stride=1, pad=1)
print("PyTorch Results")
print(conv_results_pytorch)

# Perform convolution in numpy
print("Your results")
conv_results_numpy = conv_numpy_3(input_image, conv_weights, stride=1, pad=1)
print(conv_results_numpy)

```

Now we'll do the full convolution with multiple images (batch size > 1), and multiple input channels, multiple output channels.

```python

# Perform convolution in numpy
def conv_numpy_4(image, weights, stride=1, pad=1):

    # Perform zero padding
    if pad != 0:
        image = np.pad(image, ((0, 0), (0 ,0), (pad, pad), (pad, pad)),'constant')

    # Get sizes of image array and kernel weights
    batchSize,  channelsIn, imageHeightIn, imageWidthIn = image.shape
    channelsOut, channelsIn, kernelHeight, kernelWidth = weights.shape

    # Get size of output arrays
    imageHeightOut = np.floor(1 + (imageHeightIn - kernelHeight) / stride).astype(int)
    imageWidthOut = np.floor(1 + (imageWidthIn - kernelWidth) / stride).astype(int)

    # Create output
    out = np.zeros((batchSize, channelsOut, imageHeightOut, imageWidthOut), dtype=np.float32)

    for c_batch in range(batchSize):
      for c_y in range(imageHeightOut):
        for c_x in range(imageWidthOut):
          for c_channel_out in range(channelsOut):
            for c_channel_in in range(channelsIn):
              for c_kernel_y in range(kernelHeight):
                for c_kernel_x in range(kernelWidth):
                    # TODO -- Retrieve the image pixel and the weight from the convolution
                    # Replace the two lines below
                    this_pixel_value = 1.0
                    this_weight = 1.0



                    # Multiply these together and add to the output at this position
                    out[c_batch, c_channel_out, c_y, c_x] += np.sum(this_pixel_value * this_weight)
    return out

```

```python

# Set random seed so we always get same answer
np.random.seed(1)
n_batch = 2
image_height = 4
image_width = 6
channels_in = 5
kernel_size = 3
channels_out = 2

# Create random input image
input_image= np.random.normal(size=(n_batch, channels_in, image_height, image_width))
# Create random convolution kernel weights
conv_weights = np.random.normal(size=(channels_out, channels_in, kernel_size, kernel_size))

# Perform convolution using PyTorch
conv_results_pytorch = conv_pytorch(input_image, conv_weights, stride=1, pad=1)
print("PyTorch Results")
print(conv_results_pytorch)

# Perform convolution in numpy
print("Your results")
conv_results_numpy = conv_numpy_4(input_image, conv_weights, stride=1, pad=1)
print(conv_results_numpy)

```

***


<a href="https://colab.research.google.com/github/udlbook/udlbook/blob/main/Notebooks/Chap10/10_4_Downsampling_and_Upsampling.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# **Notebook 10.4: Downsampling and Upsampling**

This notebook investigates the upsampling and downsampling methods discussed in section 10.4 of the book.

Work through the cells below, running each cell in turn. In various places you will see the words "TODO". Follow the instructions at these places and make predictions about what is going to happen or write code to complete the functions.

Contact me at udlbookmail@gmail.com if you find any mistakes or have any suggestions.

```python

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray

```

```python

# Define 4 by 4 original patch
orig_4_4 = np.array([[1, 3, 5,3 ], [6,2,0,8], [4,6,1,4], [2,8,0,3]])
print(orig_4_4)

```

```python

def downsample(x_in):
  x_out = np.zeros(( int(np.ceil(x_in.shape[0]/2)), int(np.ceil(x_in.shape[1]/2)) ))
  # TODO -- write the downsampling routine
  # Replace this line
  x_out = x_out


  return x_out

```

```python

print("Original:")
print(orig_4_4)
print("Downsampled:")
print(downsample(orig_4_4))

```

Let's try that on an image to get a feel for how it works:

```python

!wget https://raw.githubusercontent.com/udlbook/udlbook/main/Notebooks/Chap10/test_image.png

```

```python

# load the image
image = Image.open('test_image.png')
# convert image to numpy array
data = asarray(image)
data_downsample = downsample(data);

plt.figure(figsize=(5,5))
plt.imshow(data, cmap='gray')
plt.show()

plt.figure(figsize=(5,5))
plt.imshow(data_downsample, cmap='gray')
plt.show()

data_downsample2 = downsample(data_downsample)
plt.figure(figsize=(5,5))
plt.imshow(data_downsample2, cmap='gray')
plt.show()

data_downsample3 = downsample(data_downsample2)
plt.figure(figsize=(5,5))
plt.imshow(data_downsample3, cmap='gray')
plt.show()

```

```python

# Now let's try max-pooling
def maxpool(x_in):
  x_out = np.zeros(( int(np.floor(x_in.shape[0]/2)), int(np.floor(x_in.shape[1]/2)) ))
  # TODO -- write the maxpool routine
  # Replace this line
  x_out = x_out

  return x_out

```

```python

print("Original:")
print(orig_4_4)
print("Maxpooled:")
print(maxpool(orig_4_4))

```

```python

# Let's see what Rick looks like:
data_maxpool = maxpool(data);

plt.figure(figsize=(5,5))
plt.imshow(data, cmap='gray')
plt.show()

plt.figure(figsize=(5,5))
plt.imshow(data_maxpool, cmap='gray')
plt.show()

data_maxpool2 = maxpool(data_maxpool)
plt.figure(figsize=(5,5))
plt.imshow(data_maxpool2, cmap='gray')
plt.show()

data_maxpool3 = maxpool(data_maxpool2)
plt.figure(figsize=(5,5))
plt.imshow(data_maxpool3, cmap='gray')
plt.show()

```

You can see that the stripes on his shirt gradually turn to white because we keep retaining the brightest local pixels.

```python

# Finally, let's try mean pooling
def meanpool(x_in):
  x_out = np.zeros(( int(np.floor(x_in.shape[0]/2)), int(np.floor(x_in.shape[1]/2)) ))
  # TODO -- write the meanpool routine
  # Replace this line
  x_out = x_out

  return x_out

```

```python

print("Original:")
print(orig_4_4)
print("Meanpooled:")
print(meanpool(orig_4_4))

```

```python

# Let's see what Rick looks like:
data_meanpool = meanpool(data);

plt.figure(figsize=(5,5))
plt.imshow(data, cmap='gray')
plt.show()

plt.figure(figsize=(5,5))
plt.imshow(data_meanpool, cmap='gray')
plt.show()

data_meanpool2 = meanpool(data_maxpool)
plt.figure(figsize=(5,5))
plt.imshow(data_meanpool2, cmap='gray')
plt.show()

data_meanpool3 = meanpool(data_meanpool2)
plt.figure(figsize=(5,5))
plt.imshow(data_meanpool3, cmap='gray')
plt.show()

```

Notice that the three low resolution images look quite different. <br>

Now let's upscale them again

```python

# Define 2 by 2 original patch
orig_2_2 = np.array([[6, 8], [8,4]])
print(orig_2_2)

```

```python

# Let's first use the duplication method
def duplicate(x_in):
  x_out = np.zeros(( x_in.shape[0]*2, x_in.shape[1]*2 ))
  # TODO -- write the duplication routine
  # Replace this line
  x_out = x_out

  return x_out

```

```python

print("Original:")
print(orig_2_2)
print("Duplicated:")
print(duplicate(orig_2_2))

```

```python

# Let's re-upsample, downsampled rick
data_duplicate = duplicate(data_downsample3);

plt.figure(figsize=(5,5))
plt.imshow(data_downsample3, cmap='gray')
plt.show()

plt.figure(figsize=(5,5))
plt.imshow(data_duplicate, cmap='gray')
plt.show()

data_duplicate2 = duplicate(data_duplicate)
plt.figure(figsize=(5,5))
plt.imshow(data_duplicate2, cmap='gray')
plt.show()

data_duplicate3 = duplicate(data_duplicate2)
plt.figure(figsize=(5,5))
plt.imshow(data_duplicate3, cmap='gray')
plt.show()

```

They look the same, but if you look at the axes, you'll see that the pixels are just duplicated.

```python

# Now let's try max pooling back up
# The input x_high_res is the original high res image, from which you can deduce the position of the maximum index
def max_unpool(x_in, x_high_res):
  x_out = np.zeros(( x_in.shape[0]*2, x_in.shape[1]*2 ))
  # TODO -- write the unpooling routine
  # Replace this line
  x_out = x_out

  return x_out

```

```python

print("Original:")
print(orig_2_2)
print("Max unpooled:")
print(max_unpool(orig_2_2,orig_4_4))

```

```python

# Let's re-upsample, down-sampled rick
data_max_unpool= max_unpool(data_maxpool3,data_maxpool2);

plt.figure(figsize=(5,5))
plt.imshow(data_maxpool3, cmap='gray')
plt.show()

plt.figure(figsize=(5,5))
plt.imshow(data_max_unpool, cmap='gray')
plt.show()

data_max_unpool2 = max_unpool(data_max_unpool, data_maxpool)
plt.figure(figsize=(5,5))
plt.imshow(data_max_unpool2, cmap='gray')
plt.show()

data_max_unpool3 = max_unpool(data_max_unpool2, data)
plt.figure(figsize=(5,5))
plt.imshow(data_max_unpool3, cmap='gray')
plt.show()

```

Finally, we'll try upsampling using bilinear interpolation.  We'll treat the positions off the image as zeros by padding the original image and round fractional values upwards using np.ceil()

```python

def bilinear(x_in):
  x_out = np.zeros(( x_in.shape[0]*2, x_in.shape[1]*2 ))
  x_in_pad = np.zeros((x_in.shape[0]+1, x_in.shape[1]+1))
  x_in_pad[0:x_in.shape[0],0:x_in.shape[1]] = x_in
  # TODO -- write the duplication routine
  # Replace this line
  x_out = x_out

  return x_out

```

```python

print("Original:")
print(orig_2_2)
print("Bilinear:")
print(bilinear(orig_2_2))

```

```python

# Let's re-upsample, down-sampled rick
data_bilinear = bilinear(data_meanpool3);

plt.figure(figsize=(5,5))
plt.imshow(data_meanpool3, cmap='gray')
plt.show()

plt.figure(figsize=(5,5))
plt.imshow(data_bilinear, cmap='gray')
plt.show()

data_bilinear2 = bilinear(data_bilinear)
plt.figure(figsize=(5,5))
plt.imshow(data_bilinear2, cmap='gray')
plt.show()

data_bilinear3 = duplicate(data_bilinear2)
plt.figure(figsize=(5,5))
plt.imshow(data_bilinear3, cmap='gray')
plt.show()

```

***


<a href="https://colab.research.google.com/github/udlbook/udlbook/blob/main/Notebooks/Chap10/10_5_Convolution_For_MNIST.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# **Notebook 10.5: Convolution for MNIST**

This notebook builds a proper network for 2D convolution.  It works with the MNIST dataset (figure 15.15a), which was the original classic dataset for classifying images.  The network will take a 28x28 grayscale image and classify it into one of 10 classes representing a digit.

The code is adapted from https://nextjournal.com/gkoehler/pytorch-mnist

Work through the cells below, running each cell in turn. In various places you will see the words "TODO". Follow the instructions at these places and make predictions about what is going to happen or write code to complete the functions.

If you are using Google Colab, you can change your runtime to an instance with GPU support to speed up training, e.g. a T4 GPU. If you do this, the cell below should output ``device(type='cuda')``

Contact me at udlbookmail@gmail.com if you find any mistakes or have any suggestions.

```python

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

```

```python

# Run this once to load the train and test data straight into a dataloader class
# that will provide the batches

# (It may complain that some files are missing because the files seem to have been
# reorganized on the underlying website, but it still seems to work). If everything is working
# properly, then the whole notebook should run to the end without further problems
# even before you make changes.
batch_size_train = 64
batch_size_test = 1000

# TODO Change this directory to point towards an existing directory (No change needed if using Google Colab)
myDir = '/files/'

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST(myDir, train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST(myDir, train=False, download=True,
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
# TODO Change this class to implement
# 1. A valid convolution with kernel size 5, 1 input channel and 10 output channels
# 2. A max pooling operation over a 2x2 area
# 3. A Relu
# 4. A valid convolution with kernel size 5, 10 input channels and 20 output channels
# 5. A 2D Dropout layer
# 6. A max pooling operation over a 2x2 area
# 7. A relu
# 8. A flattening operation
# 9. A fully connected layer mapping from (whatever dimensions we are at-- find out using .shape) to 50
# 10. A ReLU
# 11. A fully connected layer mapping from 50 to 10 dimensions
# 12. A softmax function.

# Replace this class which implements a minimal network (which still does okay)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Valid convolution, 1 channel in, 2 channels out, stride 1, kernel size = 3
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3)
        # Dropout for convolutions
        self.drop = nn.Dropout2d()
        # Fully connected layer
        self.fc1 = nn.Linear(338, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.drop(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = x.flatten(1)
        x = self.fc1(x)
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
model = Net().to(device)
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
    data = data.to(device)
    target = target.to(device)
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
      data = data.to(device)
      target = target.to(device)
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
for i in range(10):
  plt.subplot(5,5,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Prediction: {}".format(
    output.data.max(1, keepdim=True)[1][i].item()))
  plt.xticks([])
  plt.yticks([])
plt.show()

```

***
