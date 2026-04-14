# Chap 16: 归一化流 (Normalizing Flows)

> UDLbook 精读笔记
>
> **官方资源**: [GitHub Notebooks](https://github.com/udlbook/udlbook/tree/main/Notebooks/Chap16)

---

## Notebook 列表

- **1D 归一化流**: `Chap16/16_1_1D_Normalizing_Flows.ipynb`
- **自回归流**: `Chap16/16_2_Autoregressive_Flows.ipynb`
- **收缩映射**: `Chap16/16_3_Contraction_Mappings.ipynb`

---

## 内容

<a href="https://colab.research.google.com/github/udlbook/udlbook/blob/main/Notebooks/Chap16/16_1_1D_Normalizing_Flows.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# **Notebook 16.1: 1D normalizing flows**

This notebook investigates a 1D normalizing flows example similar to that illustrated in figures 16.1 to 16.3 in the book.

Work through the cells below, running each cell in turn. In various places you will see the words "TODO". Follow the instructions at these places and make predictions about what is going to happen or write code to complete the functions.

Contact me at udlbookmail@gmail.com if you find any mistakes or have any suggestions.

```python

import numpy as np
import matplotlib.pyplot as plt

```

First we start with a base probability density function

```python

# Define the base pdf
def gauss_pdf(z, mu, sigma):
  pr_z = np.exp( -0.5 * (z-mu) * (z-mu) / (sigma * sigma))/(np.sqrt(2*3.1413) * sigma)
  return pr_z

```

```python

z = np.arange(-3,3,0.01)
pr_z = gauss_pdf(z, 0, 1)

fig,ax = plt.subplots()
ax.plot(z, pr_z)
ax.set_xlim([-3,3])
ax.set_xlabel('$z$')
ax.set_ylabel('$Pr(z)$')
plt.show();

```

Now let's define a nonlinear function that maps from the latent space $z$ to the observed data $x$.

```python

# Define a function that maps from the base pdf over z to the observed space x
def f(z):
    x1 = 6/(1+np.exp(-(z-0.25)*1.5))-3
    x2 = z
    p = z * z/9
    x = (1-p) * x1 + p * x2
    return x

# Compute gradient of that function using finite differences
def df_dz(z):
    return (f(z+0.0001)-f(z-0.0001))/0.0002

```

```python

x = f(z)
fig, ax = plt.subplots()
ax.plot(z,x)
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
ax.set_xlabel('Latent variable, $z$')
ax.set_ylabel('Observed variable, $x$')
plt.show()

```

Now let's evaluate the density in the observed space using equation 16.1

```python

# TODO -- plot the density in the observed space
# Replace these line
x = np.ones_like(z)
pr_x = np.ones_like(pr_z)

```

```python

# Plot the density in the observed space
fig,ax = plt.subplots()
ax.plot(x, pr_x)
ax.set_xlim([-3,3])
ax.set_ylim([0, 0.5])
ax.set_xlabel('$x$')
ax.set_ylabel('$Pr(x)$')
plt.show();

```

Now let's draw some samples from the new distribution (see section 16.1)

```python

np.random.seed(1)
n_sample = 20

# TODO -- Draw samples from the modeled density
# Replace this line
x_samples = np.ones((n_sample, 1))

```

```python

# Draw the samples
fig,ax = plt.subplots()
ax.plot(x, pr_x)
for x_sample in x_samples:
  ax.plot([x_sample, x_sample], [0,0.1], 'r-')

ax.set_xlim([-3,3])
ax.set_ylim([0, 0.5])
ax.set_xlabel('$x$')
ax.set_ylabel('$Pr(x)$')
plt.show();

```

***


<a href="https://colab.research.google.com/github/udlbook/udlbook/blob/main/Notebooks/Chap16/16_2_Autoregressive_Flows.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# **Notebook 16.2: 1D autoregressive flows**

This notebook investigates a 1D normalizing flows example similar to that illustrated in figure 16.7 in the book.

Work through the cells below, running each cell in turn. In various places you will see the words "TODO". Follow the instructions at these places and make predictions about what is going to happen or write code to complete the functions.

Contact me at udlbookmail@gmail.com if you find any mistakes or have any suggestions.

```python

import numpy as np
import matplotlib.pyplot as plt

```

First we'll define an invertible one dimensional function as in figure 16.5

```python

# First let's make the 1D piecewise linear mapping as illustrated in figure 16.5
def g(h, phi):
  # TODO -- write this function (equation 16.12)
  # Note: If you have the first printing of the book, there is a mistake in equation 16.12
  # Check the errata for the correct equation (or figure it out yourself!)
  # Replace this line:
  h_prime = 1


  return h_prime

```

```python

# Let's test this out.  If you managed to vectorize the routine above, then good for you
# but I'll assume you didn't and so we'll use a loop

# Define the parameters
phi = np.array([0.2, 0.1, 0.4, 0.05, 0.25])

# Run the function on an array
h = np.arange(0,1,0.01)
h_prime = np.zeros_like(h)
for i in range(len(h)):
  h_prime[i] = g(h[i], phi)

# Draw the function
fig, ax = plt.subplots()
ax.plot(h,h_prime, 'b-')
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_xlabel('Input, $h$')
ax.set_ylabel('Output, $h^\prime$')
plt.show()

```

We will also need the inverse of this function

```python

# Define the inverse function
def g_inverse(h_prime, phi):
    # Lot's of ways to do this, but we'll just do it by bracketing
    h_low = 0
    h_mid = 0.5
    h_high = 0.999

    thresh = 0.0001
    c_iter = 0
    while(c_iter < 20 and h_high - h_low > thresh):
        h_prime_low = g(h_low, phi)
        h_prime_mid = g(h_mid, phi)
        h_prime_high = g(h_high, phi)
        if h_prime_mid < h_prime:
          h_low = h_mid
        else:
          h_high = h_mid

        h_mid = h_low+(h_high-h_low)/2
        c_iter+=1

    return h_mid

```

Now let's define an autoregressive flow.  Let's switch to looking at figure 16.7.# We'll assume that our piecewise function will use five parameters phi1,phi2,phi3,phi4,phi5

```python

def ReLU(preactivation):
  activation = preactivation.clip(0.0)
  return activation

def softmax(x):
  x = np.exp(x) ;
  x = x/ np.sum(x) ;
  return x

# Return value of phi that doesn't depend on any of the inputs
def get_phi():
  return np.array([0.2, 0.1, 0.4, 0.05, 0.25])

# Compute values of phi that depend on h1
def shallow_network_phi_h1(h1, n_hidden=10):
  # For neatness of code, we'll just define the parameters in the network definition itself
  n_input = 1
  np.random.seed(n_input)
  beta0 = np.random.normal(size=(n_hidden,1))
  Omega0 = np.random.normal(size=(n_hidden, n_input))
  beta1 = np.random.normal(size=(5,1))
  Omega1 = np.random.normal(size=(5, n_hidden))
  return softmax(beta1 + Omega1 @ ReLU(beta0 + Omega0 @ np.array([[h1]])))

# Compute values of phi that depend on h1 and h2
def shallow_network_phi_h1h2(h1,h2,n_hidden=10):
  # For neatness of code, we'll just define the parameters in the network definition itself
  n_input = 2
  np.random.seed(n_input)
  beta0 = np.random.normal(size=(n_hidden,1))
  Omega0 = np.random.normal(size=(n_hidden, n_input))
  beta1 = np.random.normal(size=(5,1))
  Omega1 = np.random.normal(size=(5, n_hidden))
  return softmax(beta1 + Omega1 @ ReLU(beta0 + Omega0 @ np.array([[h1],[h2]])))

# Compute values of phi that depend on h1, h2, and h3
def shallow_network_phi_h1h2h3(h1,h2,h3, n_hidden=10):
  # For neatness of code, we'll just define the parameters in the network definition itself
  n_input = 3
  np.random.seed(n_input)
  beta0 = np.random.normal(size=(n_hidden,1))
  Omega0 = np.random.normal(size=(n_hidden, n_input))
  beta1 = np.random.normal(size=(5,1))
  Omega1 = np.random.normal(size=(5, n_hidden))
  return softmax(beta1 + Omega1 @ ReLU(beta0 + Omega0 @ np.array([[h1],[h2],[h3]])))

```

The forward mapping as shown in figure 16.7 a

```python

def forward_mapping(h1,h2,h3,h4):
  #TODO implement the forward mapping
  #Replace this line:
  h_prime1 = 0 ; h_prime2=0; h_prime3=0; h_prime4 = 0

  return h_prime1, h_prime2, h_prime3, h_prime4

```

The backward mapping as shown in figure 16.7b

```python

def backward_mapping(h1_prime,h2_prime,h3_prime,h4_prime):
  #TODO implement the backward mapping
  #Replace this line:
  h1=0; h2=0; h3=0; h4 = 0

  return h1,h2,h3,h4

```

Finally, let's make sure that the network really can be inverted

```python

# Test the network to see if it does invert correctly
h1 = 0.22; h2 = 0.41; h3 = 0.83; h4 = 0.53
print("Original h values %3.3f,%3.3f,%3.3f,%3.3f"%(h1,h2,h3,h4))
h1_prime, h2_prime, h3_prime, h4_prime = forward_mapping(h1,h2,h3,h4)
print("h_prime values %3.3f,%3.3f,%3.3f,%3.3f"%(h1_prime,h2_prime,h3_prime,h4_prime))
h1,h2,h3,h4 =  backward_mapping(h1_prime,h2_prime,h3_prime,h4_prime)
print("Reconstructed h values %3.3f,%3.3f,%3.3f,%3.3f"%(h1,h2,h3,h4))

```

***


<a href="https://colab.research.google.com/github/udlbook/udlbook/blob/main/Notebooks/Chap16/16_3_Contraction_Mappings.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# **Notebook 16.3: Contraction mappings**

This notebook investigates a 1D normalizing flows example similar to that illustrated in figure 16.9 in the book.

Work through the cells below, running each cell in turn. In various places you will see the words "TODO". Follow the instructions at these places and make predictions about what is going to happen or write code to complete the functions.

Contact me at udlbookmail@gmail.com if you find any mistakes or have any suggestions.

```python

import numpy as np
import matplotlib.pyplot as plt

```

```python

# Define a function that is a contraction mapping
def f(z):
    return 0.3 + 0.5 *z + 0.02 * np.sin(z*15)

```

```python

def draw_function(f, fixed_point=None):
  z = np.arange(0,1,0.01)
  z_prime = f(z)

  # Draw this function
  fig, ax = plt.subplots()
  ax.plot(z, z_prime,'c-')
  ax.plot([0,1],[0,1],'k--')
  if fixed_point!=None:
    ax.plot(fixed_point, fixed_point, 'ro')
  ax.set_xlim(0,1)
  ax.set_ylim(0,1)
  ax.set_xlabel('Input, $z$')
  ax.set_ylabel('Output, f$[z]$')
  plt.show()

```

```python

draw_function(f)

```

Now let's find where $\text{f}[z]=z$ using fixed point iteration

```python

# Takes a function f and a starting point z
def fixed_point_iteration(f, z0):
  # TODO -- write this function
  # Print out the iterations as you go, so you can see the progress
  # Set the maximum number of iterations to 20
  # Replace this line
  z_out = 0.5;



  return z_out

```

Now let's test that and plot the solution

```python

# Now let's test that
z = fixed_point_iteration(f, 0.2)
draw_function(f, z)

```

```python

# Let's define another function
def f2(z):
    return 0.7 + -0.6 *z + 0.03 * np.sin(z*15)
draw_function(f2)

```

```python

# Now let's test that
# TODO Before running this code, predict what you think will happen
z = fixed_point_iteration(f2, 0.9)
draw_function(f2, z)

```

```python

# Let's define another function
# Define a function that is a contraction mapping
def f3(z):
    return -0.2 + 1.5 *z + 0.1 * np.sin(z*15)
draw_function(f3)

```

```python

# Now let's test that
# TODO Before running this code, predict what you think will happen
z = fixed_point_iteration(f3, 0.7)
draw_function(f3, z)

```

Finally, let's invert a problem of the form $y = z+ f[z]$  for a given value of $y$. What is the $z$ that maps to it?

```python

def f4(z):
   return -0.3 + 0.5 *z + 0.02 * np.sin(z*15)

```

```python

def fixed_point_iteration_z_plus_f(f, y, z0):
  # TODO -- write this function
  # Replace this line
  z_out = 1

  return z_out

```

```python

def draw_function2(f, y, fixed_point=None):
  z = np.arange(0,1,0.01)
  z_prime = z+f(z)

  # Draw this function
  fig, ax = plt.subplots()
  ax.plot(z, z_prime,'c-')
  ax.plot(z, y-f(z),'r-')
  ax.plot([0,1],[0,1],'k--')
  if fixed_point!=None:
    ax.plot(fixed_point, y, 'ro')
  ax.set_xlim(0,1)
  ax.set_ylim(0,1)
  ax.set_xlabel('Input, $z$')
  ax.set_ylabel('Output, z+f$[z]$')
  plt.show()

```

```python

# Test this out and draw
y = 0.8
z = fixed_point_iteration_z_plus_f(f4,y,0.2)
draw_function2(f4,y,z)
# If you have done this correctly, the red dot should be
# where the cyan curve has a y value of 0.8

```

***
