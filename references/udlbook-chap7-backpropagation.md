# Chap 7: 反向传播 (Backpropagation)

> UDLbook 精读笔记
>
> **官方资源**: [GitHub Notebooks](https://github.com/udlbook/udlbook/tree/main/Notebooks/Chap07)

---

## Notebook 列表

- **反向传播 Toy 模型**: `Chap07/7_1_Backpropagation_in_Toy_Model.ipynb`
- **反向传播**: `Chap07/7_2_Backpropagation.ipynb`
- **初始化**: `Chap07/7_3_Initialization.ipynb`

---

## 内容

# **Notebook 7.1: Backpropagation in Toy Model**

This notebook computes the derivatives of the toy function discussed in section 7.3 of the book.

Work through the cells below, running each cell in turn. In various places you will see the words "TODO". Follow the instructions at these places and make predictions about what is going to happen or write code to complete the functions.

Contact me at udlbookmail@gmail.com if you find any mistakes or have any suggestions.

We're going to investigate how to take the derivatives of functions where one operation is composed with another, which is composed with a third and so on.  For example, consider the model:

\begin{equation}
     \text{f}[x,\boldsymbol\phi] = \beta_3+\omega_3\cdot\cos\Bigl[\beta_2+\omega_2\cdot\exp\bigl[\beta_1+\omega_1\cdot\sin[\beta_0+\omega_0x]\bigr]\Bigr],
\end{equation}

with parameters $\boldsymbol\phi=\{\beta_0,\omega_0,\beta_1,\omega_1,\beta_2,\omega_2,\beta_3,\omega_3\}$.<br>

This is a composition of the functions $\cos[\bullet],\exp[\bullet],\sin[\bullet]$.   I chose these just because you probably already know the derivatives of these functions:

\begin{align}
 \frac{\partial \cos[z]}{\partial z} = -\sin[z] \quad\quad \frac{\partial \exp[z]}{\partial z} = \exp[z] \quad\quad \frac{\partial \sin[z]}{\partial z} = \cos[z].
\end{align}

Suppose that we have a least squares loss function:

\begin{equation*}
\ell_i = (\text{f}[x_i,\boldsymbol\phi]-y_i)^2,
\end{equation*}

Assume that we know the current values of $\beta_{0},\beta_{1},\beta_{2},\beta_{3},\omega_{0},\omega_{1},\omega_{2},\omega_{3}$, $x_i$ and $y_i$. We could obviously calculate $\ell_i$.   But we also want to know how $\ell_i$ changes when we make a small change to $\beta_{0},\beta_{1},\beta_{2},\beta_{3},\omega_{0},\omega_{1},\omega_{2}$, or $\omega_{3}$.  In other words, we want to compute the eight derivatives:

\begin{align}
\frac{\partial \ell_i}{\partial \beta_{0}}, \quad \frac{\partial \ell_i}{\partial \beta_{1}}, \quad \frac{\partial \ell_i}{\partial \beta_{2}}, \quad \frac{\partial \ell_i }{\partial \beta_{3}},  \quad \frac{\partial \ell_i}{\partial \omega_{0}}, \quad \frac{\partial \ell_i}{\partial \omega_{1}}, \quad \frac{\partial \ell_i}{\partial \omega_{2}},  \quad\text{and} \quad \frac{\partial \ell_i}{\partial \omega_{3}}.
\end{align}

```python

# import library
import numpy as np

```

Let's first define the original function for $y$ and the loss term:

```python

def fn(x, beta0, beta1, beta2, beta3, omega0, omega1, omega2, omega3):
  return beta3+omega3 * np.cos(beta2 + omega2 * np.exp(beta1 + omega1 * np.sin(beta0 + omega0 * x)))

def loss(x, y, beta0, beta1, beta2, beta3, omega0, omega1, omega2, omega3):
  diff = fn(x, beta0, beta1, beta2, beta3, omega0, omega1, omega2, omega3) - y
  return diff * diff

```

Now we'll choose some values for the betas and the omegas and x and compute the output of the function:

```python

beta0 = 1.0; beta1 = 2.0; beta2 = -3.0; beta3 = 0.4
omega0 = 0.1; omega1 = -0.4; omega2 = 2.0; omega3 = 3.0
x = 2.3; y = 2.0
l_i_func = loss(x,y,beta0,beta1,beta2,beta3,omega0,omega1,omega2,omega3)
print('l_i=%3.3f'%l_i_func)

```

# Computing derivatives by hand

We could compute expressions for the derivatives by hand and write code to compute them directly but some have very complex expressions, even for this relatively simple original equation. For example:

\begin{align}
\frac{\partial \ell_i}{\partial \omega_{0}} &=& -2 \left( \beta_3+\omega_3\cdot\cos\Bigl[\beta_2+\omega_2\cdot\exp\bigl[\beta_1+\omega_1\cdot\sin[\beta_0+\omega_0\cdot x_i]\bigr]\Bigr]-y_i\right)\nonumber \\
&&\hspace{0.5cm}\cdot \omega_1\omega_2\omega_3\cdot x_i\cdot\cos[\beta_0+\omega_0 \cdot x_i]\cdot\exp\Bigl[\beta_1 + \omega_1 \cdot \sin[\beta_0+\omega_0\cdot x_i]\Bigr]\nonumber\\
&& \hspace{1cm}\cdot \sin\biggl[\beta_2+\omega_2\cdot \exp\Bigl[\beta_1 + \omega_1 \cdot \sin[\beta_0+\omega_0\cdot x_i]\Bigr]\biggr].
\end{align}

```python

dldbeta3_func = 2 * (beta3 +omega3 * np.cos(beta2 + omega2 * np.exp(beta1+omega1 * np.sin(beta0+omega0 * x)))-y)
dldomega0_func = -2 *(beta3 +omega3 * np.cos(beta2 + omega2 * np.exp(beta1+omega1 * np.sin(beta0+omega0 * x)))-y) * \
              omega1 * omega2 * omega3 * x * np.cos(beta0 + omega0 * x) * np.exp(beta1 +omega1 * np.sin(beta0 + omega0 * x)) *\
              np.sin(beta2 + omega2 * np.exp(beta1+ omega1* np.sin(beta0+omega0 * x)))

```

Let's make sure this is correct using finite differences:

```python

dldomega0_fd = (loss(x,y,beta0,beta1,beta2,beta3,omega0+0.00001,omega1,omega2,omega3)-loss(x,y,beta0,beta1,beta2,beta3,omega0,omega1,omega2,omega3))/0.00001

print('dydomega0: Function value = %3.3f, Finite difference value = %3.3f'%(dldomega0_func,dldomega0_fd))

```

The code to calculate $\partial l_i/ \partial \omega_0$ is a bit of a nightmare.  It's easy to make mistakes, and you can see that some parts of it are repeated (for example, the $\sin[\bullet]$ term), which suggests some kind of redundancy in the calculations.  The goal of this practical is to compute the derivatives in a much simpler way.  There will be three steps:

**Step 1:** Write the original equations as a series of intermediate calculations.

\begin{align}
f_{0} &=& \beta_{0} + \omega_{0} x_i\nonumber\\
h_{1} &=& \sin[f_{0}]\nonumber\\
f_{1} &=& \beta_{1} + \omega_{1}h_{1}\nonumber\\
h_{2} &=& \exp[f_{1}]\nonumber\\
f_{2} &=& \beta_{2} + \omega_{2} h_{2}\nonumber\\
h_{3} &=& \cos[f_{2}]\nonumber\\
f_{3} &=& \beta_{3} + \omega_{3}h_{3}\nonumber\\
l_i &=& (f_3-y_i)^2
\end{align}

and compute and store the values of all of these intermediate values.  We'll need them to compute the derivatives.<br>  This is called the **forward pass**.

```python

# TODO compute all the f_k and h_k terms
# Replace the code below

f0 = 0
h1 = 0
f1 = 0
h2 = 0
f2 = 0
h3 = 0
f3 = 0
l_i = 0

```

```python

# Let's check we got that right:
print("f0: true value = %3.3f, your value = %3.3f"%(1.230, f0))
print("h1: true value = %3.3f, your value = %3.3f"%(0.942, h1))
print("f1: true value = %3.3f, your value = %3.3f"%(1.623, f1))
print("h2: true value = %3.3f, your value = %3.3f"%(5.068, h2))
print("f2: true value = %3.3f, your value = %3.3f"%(7.137, f2))
print("h3: true value = %3.3f, your value = %3.3f"%(0.657, h3))
print("f3: true value = %3.3f, your value = %3.3f"%(2.372, f3))
print("l_i original = %3.3f, l_i from forward pass = %3.3f"%(l_i_func, l_i))

```

**Step 2:** Compute the derivatives of $\ell_i$ with respect to the intermediate quantities that we just calculated, but in reverse order:

\begin{align}
\quad \frac{\partial \ell_i}{\partial f_3}, \quad \frac{\partial \ell_i}{\partial h_3}, \quad \frac{\partial \ell_i}{\partial f_2}, \quad
\frac{\partial \ell_i}{\partial h_2}, \quad \frac{\partial \ell_i}{\partial f_1}, \quad \frac{\partial \ell_i}{\partial h_1},  \quad\text{and} \quad \frac{\partial \ell_i}{\partial f_0}.
\end{align}

The first of these derivatives is straightforward:

\begin{equation}
\frac{\partial \ell_i}{\partial f_{3}} = 2 (f_3-y).
\end{equation}

The second derivative can be calculated using the chain rule:

\begin{equation}
\frac{\partial \ell_i}{\partial h_{3}} =\frac{\partial f_{3}}{\partial h_{3}} \frac{\partial \ell_i}{\partial f_{3}} .
\end{equation}

The left-hand side asks how $\ell_i$ changes when $h_{3}$ changes.  The right-hand side says we can decompose this into (i) how $\ell_i$ changes when $f_{3}$ changes and how $f_{3}$ changes when $h_{3}$ changes.  So you get a chain of events happening:  $h_{3}$ changes $f_{3}$, which changes $\ell_i$, and the derivatives represent the effects of this chain.  Notice that we computed the first of these derivatives already and is  $2 (f_3-y)$. We calculated $f_{3}$ in step 1.  The second term is the derivative of $\beta_{3} + \omega_{3}h_{3}$ with respect to $h_3$ which is simply $\omega_3$.  

We can continue in this way, computing the derivatives of the output with respect to these intermediate quantities:

\begin{align}
\frac{\partial \ell_i}{\partial f_{2}} &=& \frac{\partial h_{3}}{\partial f_{2}}\left(
\frac{\partial f_{3}}{\partial h_{3}}\frac{\partial \ell_i}{\partial f_{3}} \right)
\nonumber \\
\frac{\partial \ell_i}{\partial h_{2}} &=& \frac{\partial f_{2}}{\partial h_{2}}\left(\frac{\partial h_{3}}{\partial f_{2}}\frac{\partial f_{3}}{\partial h_{3}}\frac{\partial \ell_i}{\partial f_{3}}\right)\nonumber \\
\frac{\partial \ell_i}{\partial f_{1}} &=& \frac{\partial h_{2}}{\partial f_{1}}\left( \frac{\partial f_{2}}{\partial h_{2}}\frac{\partial h_{3}}{\partial f_{2}}\frac{\partial f_{3}}{\partial h_{3}}\frac{\partial \ell_i}{\partial f_{3}} \right)\nonumber \\
\frac{\partial \ell_i}{\partial h_{1}} &=& \frac{\partial f_{1}}{\partial h_{1}}\left(\frac{\partial h_{2}}{\partial f_{1}} \frac{\partial f_{2}}{\partial h_{2}}\frac{\partial h_{3}}{\partial f_{2}}\frac{\partial f_{3}}{\partial h_{3}}\frac{\partial \ell_i}{\partial f_{3}} \right)\nonumber \\
\frac{\partial \ell_i}{\partial f_{0}} &=& \frac{\partial h_{1}}{\partial f_{0}}\left(\frac{\partial f_{1}}{\partial h_{1}}\frac{\partial h_{2}}{\partial f_{1}} \frac{\partial f_{2}}{\partial h_{2}}\frac{\partial h_{3}}{\partial f_{2}}\frac{\partial f_{3}}{\partial h_{3}}\frac{\partial \ell_i}{\partial f_{3}} \right).
\end{align}

In each case, we have already computed all of the terms except the last one in the previous step, and the last term is simple to evaluate.  This is called the **backward pass**.

```python

# TODO -- Compute the derivatives of the output with respect
# to the intermediate computations h_k and f_k (i.e, run the backward pass)
# I've done the first two for you.  You replace the code below:
dldf3 = 2* (f3 - y)
dldh3 = omega3 * dldf3
# Replace the code below
dldf2 = 1
dldh2 = 1
dldf1 = 1
dldh1 = 1
dldf0 = 1

```

```python

# Let's check we got that right
print("dldf3: true value = %3.3f, your value = %3.3f"%(0.745, dldf3))
print("dldh3: true value = %3.3f, your value = %3.3f"%(2.234, dldh3))
print("dldf2: true value = %3.3f, your value = %3.3f"%(-1.683, dldf2))
print("dldh2: true value = %3.3f, your value = %3.3f"%(-3.366, dldh2))
print("dldf1: true value = %3.3f, your value = %3.3f"%(-17.060, dldf1))
print("dldh1: true value = %3.3f, your value = %3.3f"%(6.824, dldh1))
print("dldf0: true value = %3.3f, your value = %3.3f"%(2.281, dldf0))

```

```python

# TODO -- Calculate the final derivatives with respect to the beta and omega terms

dldbeta3 = 1
dldomega3 = 1
dldbeta2 = 1
dldomega2 = 1
dldbeta1 = 1
dldomega1 = 1
dldbeta0 = 1
dldomega0 = 1

```

```python

# Let's check we got them right
print('dldbeta3: Your value = %3.3f, True value = %3.3f'%(dldbeta3, 0.745))
print('dldomega3: Your value = %3.3f, True value = %3.3f'%(dldomega3, 0.489))
print('dldbeta2: Your value = %3.3f, True value = %3.3f'%(dldbeta2, -1.683))
print('dldomega2: Your value = %3.3f, True value = %3.3f'%(dldomega2, -8.530))
print('dldbeta1: Your value = %3.3f, True value = %3.3f'%(dldbeta1, -17.060))
print('dldomega1: Your value = %3.3f, True value = %3.3f'%(dldomega1, -16.079))
print('dldbeta0: Your value = %3.3f, True value = %3.3f'%(dldbeta0, 2.281))
print('dldomega0: Your value = %3.3f, Function value = %3.3f, Finite difference value = %3.3f'%(dldomega0, dldomega0_func, dldomega0_fd))

```

Using this method, we can compute the derivatives quite easily without needing to compute very complicated expressions.  In the next practical, we'll apply this same method to a deep neural network.

***


# **Notebook 7.2: Backpropagation**

This notebook runs the backpropagation algorithm on a deep neural network as described in section 7.4 of the book.

Work through the cells below, running each cell in turn. In various places you will see the words "TODO". Follow the instructions at these places and make predictions about what is going to happen or write code to complete the functions.

Contact me at udlbookmail@gmail.com if you find any mistakes or have any suggestions.

```python

import numpy as np
import matplotlib.pyplot as plt

```

First let's define a neural network.  We'll just choose the weights and biases randomly for now

```python

# Set seed so we always get the same random numbers
np.random.seed(0)

# Number of hidden layers
K = 5
# Number of neurons per layer
D = 6
# Input layer
D_i = 1
# Output layer
D_o = 1

# Make empty lists
all_weights = [None] * (K+1)
all_biases = [None] * (K+1)

# Create input and output layers
all_weights[0] = np.random.normal(size=(D, D_i))
all_weights[-1] = np.random.normal(size=(D_o, D))
all_biases[0] = np.random.normal(size =(D,1))
all_biases[-1]= np.random.normal(size =(D_o,1))

# Create intermediate layers
for layer in range(1,K):
  all_weights[layer] = np.random.normal(size=(D,D))
  all_biases[layer] = np.random.normal(size=(D,1))

```

```python

# Define the Rectified Linear Unit (ReLU) function
def ReLU(preactivation):
  activation = preactivation.clip(0.0)
  return activation

```

Now let's run our random network.  The weight matrices $\boldsymbol\Omega_{0\ldots K}$ are the entries of the list "all_weights" and the biases $\boldsymbol\beta_{0\ldots K}$ are the entries of the list "all_biases"

We know that we will need the preactivations $\mathbf{f}_{0\ldots K}$ and the activations $\mathbf{h}_{1\ldots K}$ for the forward pass of backpropagation, so we'll store and return these as well.

```python

def compute_network_output(net_input, all_weights, all_biases):

  # Retrieve number of layers
  K = len(all_weights) -1

  # We'll store the pre-activations at each layer in a list "all_f"
  # and the activations in a second list "all_h".
  all_f = [None] * (K+1)
  all_h = [None] * (K+1)

  #For convenience, we'll set
  # all_h[0] to be the input, and all_f[K] will be the output
  all_h[0] = net_input

  # Run through the layers, calculating all_f[0...K-1] and all_h[1...K]
  for layer in range(K):
      # Update preactivations and activations at this layer according to eqn 7.17
      # Remember to use np.matmul for matrix multiplications
      # TODO -- Replace the lines below
      all_f[layer] = all_h[layer]
      all_h[layer+1] = all_f[layer]

  # Compute the output from the last hidden layer
  # TODO -- Replace the line below
  all_f[K] = np.zeros_like(all_biases[-1])

  # Retrieve the output
  net_output = all_f[K]

  return net_output, all_f, all_h

```

```python

# Define input
net_input = np.ones((D_i,1)) * 1.2
# Compute network output
net_output, all_f, all_h = compute_network_output(net_input,all_weights, all_biases)
print("True output = %3.3f, Your answer = %3.3f"%(1.907, net_output[0,0]))

```

Now let's define a loss function.  We'll just use the least squares loss function. We'll also write a function to compute dloss_doutput

```python

def least_squares_loss(net_output, y):
  return np.sum((net_output-y) * (net_output-y))

def d_loss_d_output(net_output, y):
    return 2*(net_output -y);

```

```python

y = np.ones((D_o,1)) * 20.0
loss = least_squares_loss(net_output, y)
print("y = %3.3f Loss = %3.3f"%(y, loss))

```

Now let's compute the derivatives of the network.  We already computed the forward pass.  Let's compute the backward pass.

```python

# We'll need the indicator function
def indicator_function(x):
  x_in = np.array(x)
  x_in[x_in>0] = 1
  x_in[x_in<=0] = 0
  return x_in

# Main backward pass routine
def backward_pass(all_weights, all_biases, all_f, all_h, y):
  # We'll store the derivatives dl_dweights and dl_dbiases in lists as well
  all_dl_dweights = [None] * (K+1)
  all_dl_dbiases = [None] * (K+1)
  # And we'll store the derivatives of the loss with respect to the activation and preactivations in lists
  all_dl_df = [None] * (K+1)
  all_dl_dh = [None] * (K+1)
  # Again for convenience we'll stick with the convention that all_h[0] is the net input and all_f[k] in the net output

  # Compute derivatives of the loss with respect to the network output
  all_dl_df[K] = np.array(d_loss_d_output(all_f[K],y))

  # Now work backwards through the network
  for layer in range(K,-1,-1):
    # TODO Calculate the derivatives of the loss with respect to the biases at layer from all_dl_df[layer]. (eq 7.22)
    # NOTE!  To take a copy of matrix X, use Z=np.array(X)
    # REPLACE THIS LINE
    all_dl_dbiases[layer] = np.zeros_like(all_biases[layer])

    # TODO Calculate the derivatives of the loss with respect to the weights at layer from all_dl_df[layer] and all_h[layer] (eq 7.23)
    # Don't forget to use np.matmul
    # REPLACE THIS LINE
    all_dl_dweights[layer] = np.zeros_like(all_weights[layer])

    # TODO: calculate the derivatives of the loss with respect to the activations from weight and derivatives of next preactivations (second part of last line of eq 7.25)
    # REPLACE THIS LINE
    all_dl_dh[layer] = np.zeros_like(all_h[layer])


    if layer > 0:
      # TODO Calculate the derivatives of the loss with respect to the pre-activation f (use derivative of ReLu function, first part of last line of eq. 7.25)
      # REPLACE THIS LINE
      all_dl_df[layer-1] = np.zeros_like(all_f[layer-1])

  return all_dl_dweights, all_dl_dbiases

```

```python

all_dl_dweights, all_dl_dbiases = backward_pass(all_weights, all_biases, all_f, all_h, y)

```

```python

np.set_printoptions(precision=3)
# Make space for derivatives computed by finite differences
all_dl_dweights_fd = [None] * (K+1)
all_dl_dbiases_fd = [None] * (K+1)

# Let's test if we have the derivatives right using finite differences
delta_fd = 0.000001

# Test the dervatives of the bias vectors
for layer in range(K+1):
  dl_dbias  = np.zeros_like(all_dl_dbiases[layer])
  # For every element in the bias
  for row in range(all_biases[layer].shape[0]):
    # Take copy of biases  We'll change one element each time
    all_biases_copy = [np.array(x) for x in all_biases]
    all_biases_copy[layer][row] += delta_fd
    network_output_1, *_ = compute_network_output(net_input, all_weights, all_biases_copy)
    network_output_2, *_ = compute_network_output(net_input, all_weights, all_biases)
    dl_dbias[row] = (least_squares_loss(network_output_1, y) - least_squares_loss(network_output_2,y))/delta_fd
  all_dl_dbiases_fd[layer] = np.array(dl_dbias)
  print("-----------------------------------------------")
  print("Bias %d, derivatives from backprop:"%(layer))
  print(all_dl_dbiases[layer])
  print("Bias %d, derivatives from finite differences"%(layer))
  print(all_dl_dbiases_fd[layer])
  if np.allclose(all_dl_dbiases_fd[layer],all_dl_dbiases[layer],rtol=1e-05, atol=1e-08, equal_nan=False):
    print("Success!  Derivatives match.")
  else:
    print("Failure!  Derivatives different.")



# Test the derivatives of the weights matrices
for layer in range(K+1):
  dl_dweight  = np.zeros_like(all_dl_dweights[layer])
  # For every element in the bias
  for row in range(all_weights[layer].shape[0]):
    for col in range(all_weights[layer].shape[1]):
      # Take copy of biases  We'll change one element each time
      all_weights_copy = [np.array(x) for x in all_weights]
      all_weights_copy[layer][row][col] += delta_fd
      network_output_1, *_ = compute_network_output(net_input, all_weights_copy, all_biases)
      network_output_2, *_ = compute_network_output(net_input, all_weights, all_biases)
      dl_dweight[row][col] = (least_squares_loss(network_output_1, y) - least_squares_loss(network_output_2,y))/delta_fd
  all_dl_dweights_fd[layer] = np.array(dl_dweight)
  print("-----------------------------------------------")
  print("Weight %d, derivatives from backprop:"%(layer))
  print(all_dl_dweights[layer])
  print("Weight %d, derivatives from finite differences"%(layer))
  print(all_dl_dweights_fd[layer])
  if np.allclose(all_dl_dweights_fd[layer],all_dl_dweights[layer],rtol=1e-05, atol=1e-08, equal_nan=False):
    print("Success!  Derivatives match.")
  else:
    print("Failure!  Derivatives different.")

```

***


# **Notebook 7.3: Initialization**

This notebook explores weight initialization in deep neural networks as described in section 7.5 of the book.

Work through the cells below, running each cell in turn. In various places you will see the words "TODO". Follow the instructions at these places and make predictions about what is going to happen or write code to complete the functions.

Contact me at udlbookmail@gmail.com if you find any mistakes or have any suggestions.

```python

import numpy as np
import matplotlib.pyplot as plt

```

First let's define a neural network.  We'll just choose the weights and biases randomly for now

```python

def init_params(K, D, sigma_sq_omega):
  # Set seed so we always get the same random numbers
  np.random.seed(0)

  # Input layer
  D_i = 1
  # Output layer
  D_o = 1

  # Make empty lists
  all_weights = [None] * (K+1)
  all_biases = [None] * (K+1)

  # Create input and output layers
  all_weights[0] = np.random.normal(size=(D, D_i))*np.sqrt(sigma_sq_omega)
  all_weights[-1] = np.random.normal(size=(D_o, D)) * np.sqrt(sigma_sq_omega)
  all_biases[0] = np.zeros((D,1))
  all_biases[-1]= np.zeros((D_o,1))

  # Create intermediate layers
  for layer in range(1,K):
    all_weights[layer] = np.random.normal(size=(D,D))*np.sqrt(sigma_sq_omega)
    all_biases[layer] = np.zeros((D,1))

  return all_weights, all_biases

```

```python

# Define the Rectified Linear Unit (ReLU) function
def ReLU(preactivation):
  activation = preactivation.clip(0.0)
  return activation

```

```python

def compute_network_output(net_input, all_weights, all_biases):

  # Retrieve number of layers
  K = len(all_weights)-1

  # We'll store the pre-activations at each layer in a list "all_f"
  # and the activations in a second list "all_h".
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

Now let's investigate how the size of the outputs vary as we change the initialization variance:

```python

# Number of layers
K = 5
# Number of neurons per layer
D = 8
# Input layer
D_i = 1
# Output layer
D_o = 1
# Set variance of initial weights to 1
sigma_sq_omega = 1.0
# Initialize parameters
all_weights, all_biases = init_params(K,D,sigma_sq_omega)

n_data = 1000
data_in = np.random.normal(size=(1,n_data))
net_output, all_f, all_h = compute_network_output(data_in, all_weights, all_biases)

for layer in range(1,K+1):
  print("Layer %d, std of hidden units = %3.3f"%(layer, np.std(all_h[layer])))

```

```python

# You can see that the values of the hidden units are increasing on average (the variance is across all hidden units at the layer
# and the 1000 training examples

# TODO
# Change this to 50 layers with 80 hidden units per layer

# TODO
# Now experiment with sigma_sq_omega to try to stop the variance of the forward computation exploding

```

Now let's define a loss function.  We'll just use the least squares loss function. We'll also write a function to compute dloss_doutput

```python

def least_squares_loss(net_output, y):
  return np.sum((net_output-y) * (net_output-y))

def d_loss_d_output(net_output, y):
    return 2*(net_output -y);

```

Here's the code for the backward pass

```python

# We'll need the indicator function
def indicator_function(x):
  x_in = np.array(x)
  x_in[x_in>=0] = 1
  x_in[x_in<0] = 0
  return x_in

# Main backward pass routine
def backward_pass(all_weights, all_biases, all_f, all_h, y):
  # Retrieve number of layers
  K = len(all_weights) - 1

  # We'll store the derivatives dl_dweights and dl_dbiases in lists as well
  all_dl_dweights = [None] * (K+1)
  all_dl_dbiases = [None] * (K+1)
  # And we'll store the derivatives of the loss with respect to the activation and preactivations in lists
  all_dl_df = [None] * (K+1)
  all_dl_dh = [None] * (K+1)
  # Again for convenience we'll stick with the convention that all_h[0] is the net input and all_f[k] in the net output

  # Compute derivatives of net output with respect to loss
  all_dl_df[K] = np.array(d_loss_d_output(all_f[K],y))

  # Now work backwards through the network
  for layer in range(K,-1,-1):
    # Calculate the derivatives of biases at layer from all_dl_df[K]. (eq 7.13, line 1)
    all_dl_dbiases[layer] = np.array(all_dl_df[layer])
    # Calculate the derivatives of weight at layer from all_dl_df[K] and all_h[K] (eq 7.13, line 2)
    all_dl_dweights[layer] = np.matmul(all_dl_df[layer], all_h[layer].transpose())

    # Calculate the derivatives of activations from weight and derivatives of next preactivations (eq 7.13, line 3 second part)
    all_dl_dh[layer] = np.matmul(all_weights[layer].transpose(), all_dl_df[layer])
    # Calculate the derivatives of the pre-activation f with respect to activation h (eq 7.13, line 3, first part)
    if layer > 0:
      all_dl_df[layer-1] = indicator_function(all_f[layer-1]) * all_dl_dh[layer]

  return all_dl_dweights, all_dl_dbiases, all_dl_dh, all_dl_df

```

Now let's look at what happens to the magnitude of the gradients on the way back.

```python

# Number of layers
K = 5
# Number of neurons per layer
D = 8
# Input layer
D_i = 1
# Output layer
D_o = 1
# Set variance of initial weights to 1
sigma_sq_omega = 1.0
# Initialize parameters
all_weights, all_biases = init_params(K,D,sigma_sq_omega)

# For simplicity we'll just consider the gradients of the weights and biases between the first and last hidden layer
n_data = 100
aggregate_dl_df = [None] * (K+1)
for layer in range(1,K):
  # These 3D arrays will store the gradients for every data point
  aggregate_dl_df[layer] = np.zeros((D,n_data))


# We'll have to compute the derivatives of the parameters for each data point separately
for c_data in range(n_data):
  data_in = np.random.normal(size=(1,1))
  y = np.zeros((1,1))
  net_output, all_f, all_h = compute_network_output(data_in, all_weights, all_biases)
  all_dl_dweights, all_dl_dbiases, all_dl_dh, all_dl_df = backward_pass(all_weights, all_biases, all_f, all_h, y)
  for layer in range(1,K):
    aggregate_dl_df[layer][:,c_data] = np.squeeze(all_dl_df[layer])

for layer in reversed(range(1,K)):
  print("Layer %d, std of dl_dh = %3.3f"%(layer, np.std(aggregate_dl_df[layer].ravel())))

```

```python

# You can see that the gradients of the hidden units are increasing on average (the standard deviation is across all hidden units at the layer
# and the 100 training examples

# TODO
# Change this to 50 layers with 80 hidden units per layer

# TODO
# Now experiment with sigma_sq_omega to try to stop the variance of the gradients exploding

```

***
