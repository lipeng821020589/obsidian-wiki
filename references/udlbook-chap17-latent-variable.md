# Chap 17: 潜变量模型 (Latent Variable Models)

> UDLbook 精读笔记
>
> **官方资源**: [GitHub Notebooks](https://github.com/udlbook/udlbook/tree/main/Notebooks/Chap17)

---

## Notebook 列表

- **潜变量模型**: `Chap17/17_1_Latent_Variable_Models.ipynb`
- **重参数化技巧**: `Chap17/17_2_Reparameterization_Trick.ipynb`
- **重要性采样**: `Chap17/17_3_Importance_Sampling.ipynb`

---

## 内容

# **Notebook 17.1: Latent variable models**

This notebook investigates a non-linear latent variable model similar to that in figures 17.2 and 17.3 of the book.

Work through the cells below, running each cell in turn. In various places you will see the words "TODO". Follow the instructions at these places and make predictions about what is going to happen or write code to complete the functions.

Contact me at udlbookmail@gmail.com if you find any mistakes or have any suggestions.

```python

import numpy as np
import matplotlib.pyplot as plt
import scipy
from matplotlib.colors import ListedColormap
from matplotlib import cm

```

We'll assume that our base distribution over the latent variables is a 1D standard normal so that

\begin{equation}
Pr(z) = \text{Norm}_{z}[0,1]
\end{equation}

As in figure 17.2, we'll assume that the output is two dimensional, we need to define a function that maps from the 1D latent variable to two dimensions.  Usually, we would use a neural network, but in this case, we'll just define an arbitrary relationship.

\begin{align}
x_{1} &=& 0.5\cdot\exp\Bigl[\sin\bigl[2+ 3.675 z \bigr]\Bigr]\\
x_{2} &=& \sin\bigl[2+ 2.85 z \bigr]
\end{align}

```python

# The function that maps z to x1 and x2
def f(z):
  x_1 = np.exp(np.sin(2+z*3.675)) * 0.5
  x_2 = np.cos(2+z*2.85)
  return x_1, x_2

```

Let's plot the 3D relation between the two observed variables $x_{1}$ and $x_{2}$ and the latent variables $z$ as in figure 17.2 of the book.  We'll use the opacity to represent the prior probability $Pr(z)$.

```python

def draw_3d_projection(z,pr_z, x1,x2):
  alpha = pr_z / np.max(pr_z)
  ax = plt.axes(projection='3d')
  fig = plt.gcf()
  fig.set_size_inches(18.5, 10.5)
  for i in range(len(z)-1):
    ax.plot([z[i],z[i+1]],[x1[i],x1[i+1]],[x2[i],x2[i+1]],'r-', alpha=pr_z[i])
  ax.set_xlabel('$z$',)
  ax.set_ylabel('$x_1$')
  ax.set_zlabel('$x_2$')
  ax.set_xlim(-3,3)
  ax.set_ylim(0,2)
  ax.set_zlim(-1,1)
  ax.set_box_aspect((3,1,1))
  plt.show()

```

```python

# Compute the prior
def get_prior(z):
  return scipy.stats.multivariate_normal.pdf(z)

```

```python

# Define the latent variable values
z = np.arange(-3.0,3.0,0.01)
# Find the probability distribution over z
pr_z = get_prior(z)
# Compute x1 and x2 for each z
x1,x2 = f(z)
# Plot the function
draw_3d_projection(z,pr_z, x1,x2)

```

The likelihood is defined as:
\begin{align}
 Pr(x_1,x_2|z) &=&  \text{Norm}_{[x_1,x_2]}\Bigl[\mathbf{f}[z],\sigma^{2}\mathbf{I}\Bigr]
\end{align}

so we will also need to define the noise level $\sigma^2$

```python

sigma_sq = 0.04

```

```python

# Draws a heatmap to represent a probability distribution, possibly with samples overlaed
def plot_heatmap(x1_mesh,x2_mesh,y_mesh, x1_samples=None, x2_samples=None, title=None):
  # Define pretty colormap
  my_colormap_vals_hex =('2a0902', '2b0a03', '2c0b04', '2d0c05', '2e0c06', '2f0d07', '300d08', '310e09', '320f0a', '330f0b', '34100b', '35110c', '36110d', '37120e', '38120f', '39130f', '3a1410', '3b1411', '3c1511', '3d1612', '3e1613', '3f1713', '401714', '411814', '421915', '431915', '451a16', '461b16', '471b17', '481c17', '491d18', '4a1d18', '4b1e19', '4c1f19', '4d1f1a', '4e201b', '50211b', '51211c', '52221c', '53231d', '54231d', '55241e', '56251e', '57261f', '58261f', '592720', '5b2821', '5c2821', '5d2922', '5e2a22', '5f2b23', '602b23', '612c24', '622d25', '632e25', '652e26', '662f26', '673027', '683027', '693128', '6a3229', '6b3329', '6c342a', '6d342a', '6f352b', '70362c', '71372c', '72372d', '73382e', '74392e', '753a2f', '763a2f', '773b30', '783c31', '7a3d31', '7b3e32', '7c3e33', '7d3f33', '7e4034', '7f4134', '804235', '814236', '824336', '834437', '854538', '864638', '874739', '88473a', '89483a', '8a493b', '8b4a3c', '8c4b3c', '8d4c3d', '8e4c3e', '8f4d3f', '904e3f', '924f40', '935041', '945141', '955242', '965343', '975343', '985444', '995545', '9a5646', '9b5746', '9c5847', '9d5948', '9e5a49', '9f5a49', 'a05b4a', 'a15c4b', 'a35d4b', 'a45e4c', 'a55f4d', 'a6604e', 'a7614e', 'a8624f', 'a96350', 'aa6451', 'ab6552', 'ac6552', 'ad6653', 'ae6754', 'af6855', 'b06955', 'b16a56', 'b26b57', 'b36c58', 'b46d59', 'b56e59', 'b66f5a', 'b7705b', 'b8715c', 'b9725d', 'ba735d', 'bb745e', 'bc755f', 'bd7660', 'be7761', 'bf7862', 'c07962', 'c17a63', 'c27b64', 'c27c65', 'c37d66', 'c47e67', 'c57f68', 'c68068', 'c78169', 'c8826a', 'c9836b', 'ca846c', 'cb856d', 'cc866e', 'cd876f', 'ce886f', 'ce8970', 'cf8a71', 'd08b72', 'd18c73', 'd28d74', 'd38e75', 'd48f76', 'd59077', 'd59178', 'd69279', 'd7937a', 'd8957b', 'd9967b', 'da977c', 'da987d', 'db997e', 'dc9a7f', 'dd9b80', 'de9c81', 'de9d82', 'df9e83', 'e09f84', 'e1a185', 'e2a286', 'e2a387', 'e3a488', 'e4a589', 'e5a68a', 'e5a78b', 'e6a88c', 'e7aa8d', 'e7ab8e', 'e8ac8f', 'e9ad90', 'eaae91', 'eaaf92', 'ebb093', 'ecb295', 'ecb396', 'edb497', 'eeb598', 'eeb699', 'efb79a', 'efb99b', 'f0ba9c', 'f1bb9d', 'f1bc9e', 'f2bd9f', 'f2bfa1', 'f3c0a2', 'f3c1a3', 'f4c2a4', 'f5c3a5', 'f5c5a6', 'f6c6a7', 'f6c7a8', 'f7c8aa', 'f7c9ab', 'f8cbac', 'f8ccad', 'f8cdae', 'f9ceb0', 'f9d0b1', 'fad1b2', 'fad2b3', 'fbd3b4', 'fbd5b6', 'fbd6b7', 'fcd7b8', 'fcd8b9', 'fcdaba', 'fddbbc', 'fddcbd', 'fddebe', 'fddfbf', 'fee0c1', 'fee1c2', 'fee3c3', 'fee4c5', 'ffe5c6', 'ffe7c7', 'ffe8c9', 'ffe9ca', 'ffebcb', 'ffeccd', 'ffedce', 'ffefcf', 'fff0d1', 'fff2d2', 'fff3d3', 'fff4d5', 'fff6d6', 'fff7d8', 'fff8d9', 'fffada', 'fffbdc', 'fffcdd', 'fffedf', 'ffffe0')
  my_colormap_vals_dec = np.array([int(element,base=16) for element in my_colormap_vals_hex])
  r = np.floor(my_colormap_vals_dec/(256*256))
  g = np.floor((my_colormap_vals_dec - r *256 *256)/256)
  b = np.floor(my_colormap_vals_dec - r * 256 *256 - g * 256)
  my_colormap = ListedColormap(np.vstack((r,g,b)).transpose()/255.0)

  fig,ax = plt.subplots()
  fig.set_size_inches(8,8)
  ax.contourf(x1_mesh,x2_mesh,y_mesh,256,cmap=my_colormap)
  ax.contour(x1_mesh,x2_mesh,y_mesh,8,colors=['#80808080'])
  if title is not None:
    ax.set_title(title);
  if x1_samples is not None:
    ax.plot(x1_samples, x2_samples, 'c.')
  ax.set_xlim([-0.5,2.5])
  ax.set_ylim([-1.5,1.5])
  ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
  plt.show()

```

```python

# Returns the likelihood
def get_likelihood(x1_mesh, x2_mesh, z_val):
  # Find the corresponding x1 and x2 values
  x1,x2 = f(z_val)

  # Calculate the probability for a mesh of x1,x2 values.
  mn = scipy.stats.multivariate_normal([x1, x2], [[sigma_sq, 0], [0, sigma_sq]])
  pr_x1_x2_given_z_val = mn.pdf(np.dstack((x1_mesh, x2_mesh)))
  return pr_x1_x2_given_z_val

```

Now let's plot the likelihood $Pr(x_1,x_2|z)$ as in fig 17.3b in the book.

```python

# Choose some z value
z_val = 1.8

# Compute the conditional distribution on a grid
x1_mesh, x2_mesh = np.meshgrid(np.arange(-0.5,2.5,0.01), np.arange(-1.5,1.5,0.01))
pr_x1_x2_given_z_val = get_likelihood(x1_mesh,x2_mesh, z_val)

# Plot the result
plot_heatmap(x1_mesh, x2_mesh, pr_x1_x2_given_z_val, title="Conditional distribution $Pr(x_1,x_2|z)$")

# TODO -- Experiment with different values of z and make sure that you understand the what is happening.

```

The data density is found by marginalizing over the latent variables $z$:

\begin{align}
 Pr(x_1,x_2) &=& \int Pr(x_1,x_2, z) dz \nonumber \\
 &=& \int Pr(x_1,x_2 | z) \cdot Pr(z)dz\nonumber \\
 &=& \int \text{Norm}_{[x_1,x_2]}\Bigl[\mathbf{f}[z],\sigma^{2}\mathbf{I}\Bigr]\cdot \text{Norm}_{z}\left[\mathbf{0},\mathbf{I}\right]dz.
\end{align}

```python

# TODO Compute the data density
# We can't integrate this function in closed form
# So let's approximate it as a sum over the z values (z = np.arange(-3,3,0.01))
# You will need the functions get_likelihood() and get_prior()
# To make this a valid probability distribution, you need to multiply
# By the z-increment (0.01)
# Replace this line
pr_x1_x2 = np.zeros_like(x1_mesh)


# Plot the result
plot_heatmap(x1_mesh, x2_mesh, pr_x1_x2, title="Data density $Pr(x_1,x_2)$")

```

Now let's draw some samples from the model

```python

def draw_samples(n_sample):
  # TODO Write this routine to draw n_sample samples from the model
  # First draw a random value of z from the prior (a standard normal distribution)
  # Then draw a sample from Pr(x1,x2|z)
  # Replace this line
  x1_samples=0; x2_samples = 0;

  return x1_samples, x2_samples

```

Let's plot those samples on top of the heat map.

```python

x1_samples, x2_samples = draw_samples(500)
# Plot the result
plot_heatmap(x1_mesh, x2_mesh, pr_x1_x2, x1_samples, x2_samples, title="Data density $Pr(x_1,x_2)$")

```

```python

# Return the posterior distribution
def get_posterior(x1,x2):
  z = np.arange(-3,3, 0.01)
  # TODO -- write this function
  # Again, we can't integrate, but we can sum
  # We don't know the constant in the denominator of equation 17.19, but we can just normalize
  # by the sum of the numerators for all values of z
  # Replace this line:
  pr_z_given_x1_x2 = np.ones_like(z)


  return z, pr_z_given_x1_x2

```

```python

x1 = 0.9; x2 = -0.9
z, pr_z_given_x1_x2 = get_posterior(x1,x2)


fig, ax = plt.subplots()
ax.plot(z, pr_z_given_x1_x2, 'r-')
ax.set_xlabel("Latent variable $z$")
ax.set_ylabel("Posterior probability $Pr(z|x_{1},x_{2})$")
ax.set_xlim([-3,3])
ax.set_ylim([0,1.5 * np.max(pr_z_given_x1_x2)])
plt.show()

```

***


# **Notebook 17.2: Reparameterization trick**

This notebook investigates the reparameterization trick as described in section 17.7 of the book.

Work through the cells below, running each cell in turn. In various places you will see the words "TODO". Follow the instructions at these places and make predictions about what is going to happen or write code to complete the functions.

Contact me at udlbookmail@gmail.com if you find any mistakes or have any suggestions.

```python

import numpy as np
import matplotlib.pyplot as plt

```

The reparameterization trick computes the derivative of an expectation of a function $\text{f}[x]$:

\begin{equation}
\frac{\partial}{\partial \boldsymbol\phi} \mathbb{E}_{Pr(x|\boldsymbol\phi)}\bigl[\text{f}[x]\bigr],
\end{equation}

with respect to the parameters $\boldsymbol\phi$ of the distribution $Pr(x|\boldsymbol\phi)$ that the expectation is over.

Let's consider a simple concrete example, where:

\begin{equation}
Pr(x|\phi) = \text{Norm}_{x}\Bigl[\mu, \sigma^2\Bigr]=\text{Norm}_{x}\Bigl[\phi^3,(\exp[\phi])^2\Bigr]
\end{equation}

and

\begin{equation}
\text{f}[x] = x^2+\sin[x]
\end{equation}

```python

# Let's approximate this expectation for a particular value of phi
def compute_expectation(phi, n_samples):
  # TODO complete this function
  # 1. Compute the mean of the normal distribution, mu
  # 2. Compute the standard deviation of the normal distribution, sigma
  # 3. Draw n_samples samples using np.random.normal(mu, sigma, size=(n_samples, 1))
  # 4. Compute f[x] for each of these samples
  # 4. Approximate the expectation by taking the average of the values of f[x]
  # Replace this line
  expected_f_given_phi = 0


  return expected_f_given_phi

```

```python

# Set the seed so the random numbers are all the same
np.random.seed(0)

# Compute the expectation for two values of phi
phi1 = 0.5
n_samples = 10000000
expected_f_given_phi1 = compute_expectation(phi1, n_samples)
print("Your value: ", expected_f_given_phi1, ", True value:  2.7650801613563116")

phi2 = -0.1
n_samples = 10000000
expected_f_given_phi2 = compute_expectation(phi2, n_samples)
print("Your value: ", expected_f_given_phi2, ", True value:  0.8176793102849222")

```

Le't plot this expectation as a function of phi

```python

phi_vals = np.arange(-1.5,1.5, 0.05)
expected_vals = np.zeros_like(phi_vals)
n_samples = 1000000
for i in range(len(phi_vals)):
  expected_vals[i] = compute_expectation(phi_vals[i], n_samples)

fig,ax = plt.subplots()
ax.plot(phi_vals, expected_vals,'r-')
ax.set_xlabel(r'Parameter $\phi$')
ax.set_ylabel(r'$\mathbb{E}_{Pr(x|\phi)}[f[x]]$')
plt.show()

```

It's this curve that we want to find the derivative of (so for example, we could run gradient descent and find the minimum.

This is tricky though -- if you look at the computation that you performed, then there is a sampling step in the procedure (step 3).  How do we compute the derivative of this?

The answer is the reparameterization trick.  We note that:

\begin{equation}
\text{Norm}_{x}\Bigl[\mu, \sigma^2\Bigr]=\text{Norm}_{x}\Bigl[0, 1\Bigr] \times \sigma + \mu
\end{equation}

and so:

\begin{equation}
\text{Norm}_{x}\Bigl[\phi^3,(\exp[\phi])^2\Bigr]  = \text{Norm}_{x}\Bigl[0, 1\Bigr] \times \exp[\phi]+ \phi^3
\end{equation}

So, if we draw a sample $\epsilon^*$ from $\text{Norm}_{\epsilon}[0, 1]$, then we can compute a sample $x^*$ as:

\begin{align}
x^* &=& \epsilon^* \times \sigma + \mu \\
&=& \epsilon^* \times \exp[\phi]+ \phi^3
\end{align}

```python

def compute_df_dx_star(x_star):
  # TODO Compute this derivative (function defined at the top)
  # Replace this line:
  deriv = 0;



  return deriv

def compute_dx_star_dphi(epsilon_star, phi):
  # TODO Compute this derivative
    # Replace this line:
  deriv = 0;



  return deriv

def compute_derivative_of_expectation(phi, n_samples):
  # Generate the random values of epsilon
  epsilon_star= np.random.normal(size=(n_samples,1))
  # TODO -- write
  # 1. Compute dx*/dphi using the function defined above
  # 2. Compute x*
  # 3. Compute df/dx* using the function you wrote above
  # 4. Compute df/dphi = df/x* * dx*dphi
  # 5. Average the samples of df/dphi to get the expectation.
  # Replace this line:
  df_dphi = 0



  return df_dphi

```

```python

# Set the seed so the random numbers are all the same
np.random.seed(0)

# Compute the expectation for two values of phi
phi1 = 0.5
n_samples = 10000000

deriv = compute_derivative_of_expectation(phi1, n_samples)
print("Your value: ", deriv, ", True value:  5.726338035051403")

```

```python

phi_vals = np.arange(-1.5,1.5, 0.05)
deriv_vals = np.zeros_like(phi_vals)
n_samples = 1000000
for i in range(len(phi_vals)):
  deriv_vals[i] = compute_derivative_of_expectation(phi_vals[i], n_samples)

fig,ax = plt.subplots()
ax.plot(phi_vals, deriv_vals,'r-')
ax.set_xlabel(r'Parameter $\phi$')
ax.set_ylabel(r'$\partial/\partial\phi\mathbb{E}_{Pr(x|\phi)}[f[x]]$')
plt.show()

```

This should look plausibly like the derivative of the function we plotted above!

The reparameterization trick computes the derivative of an expectation of a function $\text{f}[x]$:

\begin{equation}
\frac{\partial}{\partial \boldsymbol\phi} \mathbb{E}_{Pr(x|\boldsymbol\phi)}\bigl[\text{f}[x]\bigr],
\end{equation}

with respect to the parameters $\boldsymbol\phi$ of the distribution $Pr(x|\boldsymbol\phi)$ that the expectation is over. This derivative can also be computed as:

\begin{align}
\frac{\partial}{\partial \boldsymbol\phi} \mathbb{E}_{Pr(x|\boldsymbol\phi)}\bigl[\text{f}[x]\bigr] &=& \mathbb{E}_{Pr(x|\boldsymbol\phi)}\left[\text{f}[x]\frac{\partial}{\partial \boldsymbol\phi} \log\bigl[ Pr(x|\boldsymbol\phi)\bigr]\right]\nonumber \\
&\approx & \frac{1}{I}\sum_{i=1}^{I}\text{f}[x_i]\frac{\partial}{\partial \boldsymbol\phi} \log\bigl[ Pr(x_i|\boldsymbol\phi)\bigr].
\end{align}

This method is known as the REINFORCE algorithm or score function estimator.  Problem 17.5 asks you to prove this relation.  Let's use this method to compute the gradient and compare.

Recall that the expression for a univariate Gaussian is:

\begin{equation}
 Pr(x|\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^{2}}}\exp\left[-\frac{(x-\mu)^{2}}{2\sigma^{2}}\right].
\end{equation}

```python

def d_log_pr_x_given_phi(x,phi):
  # TODO -- fill in this function
  # Compute the derivative of log[Pr(x|phi)]
  # Replace this line:
  deriv =0;


  return deriv


def compute_derivative_of_expectation_score_function(phi, n_samples):
  # TODO -- Compute this function
  # 1. Calculate mu from phi
  # 2. Calculate sigma from phi
  # 3. Generate n_sample random samples of x using np.random.normal
  # 4. Calculate f[x] for all of the samples
  # 5. Multiply f[x] by d_log_pr_x_given_phi
  # 6. Take the average of the samples
  # Replace this line:
  deriv = 0



  return deriv

```

```python

# Set the seed so the random numbers are all the same
np.random.seed(0)

# Compute the expectation for two values of phi
phi1 = 0.5
n_samples = 100000000

deriv = compute_derivative_of_expectation_score_function(phi1, n_samples)
print("Your value: ", deriv, ", True value:  5.724609927313369")

```

```python

phi_vals = np.arange(-1.5,1.5, 0.05)
deriv_vals = np.zeros_like(phi_vals)
n_samples = 1000000
for i in range(len(phi_vals)):
  deriv_vals[i] = compute_derivative_of_expectation_score_function(phi_vals[i], n_samples)

fig,ax = plt.subplots()
ax.plot(phi_vals, deriv_vals,'r-')
ax.set_xlabel(r'Parameter $\phi$')
ax.set_ylabel(r'$\partial/\partial\phi\mathbb{E}_{Pr(x|\phi)}[f[x]]$')
plt.show()

```

This should look the same as the derivative that we computed with the reparameterization trick.  So, is there any advantage to one way or the other?  Let's compare the variances of the estimates

```python

n_estimate = 100
n_sample = 1000
phi = 0.3
reparam_estimates = np.zeros((n_estimate,1))
score_function_estimates = np.zeros((n_estimate,1))
for i in range(n_estimate):
  reparam_estimates[i]= compute_derivative_of_expectation(phi, n_samples)
  score_function_estimates[i] = compute_derivative_of_expectation_score_function(phi, n_samples)

print("Variance of reparameterization estimator", np.var(reparam_estimates))
print("Variance of score function estimator", np.var(score_function_estimates))

```

The variance of the reparameterization estimator should be quite a bit lower than the score function estimator which is why it is preferred in this situation.

***


# **Notebook 17.3: Importance sampling**

This notebook investigates importance sampling as described in section 17.8.1 of the book.

Work through the cells below, running each cell in turn. In various places you will see the words "TODO". Follow the instructions at these places and make predictions about what is going to happen or write code to complete the functions.

Contact me at udlbookmail@gmail.com if you find any mistakes or have any suggestions.

```python

import numpy as np
import matplotlib.pyplot as plt

```

Let's approximate the expectation

\begin{equation}
\mathbb{E}_{y}\Bigl[\exp\bigl[- (y-1)^4\bigr]\Bigr] = \int \exp\bigl[- (y-1)^4\bigr] Pr(y) dy,
\end{equation}

where

\begin{equation}
Pr(y)=\text{Norm}_y[0,1]
\end{equation}

by drawing $I$ samples $y_i$ and using the formula:

\begin{equation}
\mathbb{E}_{y}\Bigl[\exp\bigl[- (y-1)^4\bigr]\Bigr] \approx \frac{1}{I} \sum_{i=1}^I \exp\bigl[-(y_i-1)^4 \bigr]
\end{equation}

```python

def f(y):
  return np.exp(-(y-1) *(y-1) *(y-1) * (y-1))


def pr_y(y):
  return (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * y * y)

fig,ax = plt.subplots()
y = np.arange(-10,10,0.01)
ax.plot(y, f(y), 'r-', label='f$[y]$');
ax.plot(y, pr_y(y),'b-',label='$Pr(y)$')
ax.set_xlabel("$y$")
ax.legend()
plt.show()

```

```python

def compute_expectation(n_samples):
  # TODO -- compute this expectation
  # 1. Generate samples y_i using np.random.normal
  # 2. Approximate the expectation of f[y]
  # Replace this line
  expectation = 0


  return expectation

```

```python

# Set the seed so the random numbers are all the same
np.random.seed(0)

# Compute the expectation  with a very large number of samples (good estimate)
n_samples = 100000000
expected_f= compute_expectation(n_samples)
print("Your value: ", expected_f, ", True value:  0.43160702267383166")

```

Let's investigate how the variance of this approximation decreases as we increase the number of samples $N$.

```python

def compute_mean_variance(n_sample):
  n_estimate = 10000
  estimates = np.zeros((n_estimate,1))
  for i in range(n_estimate):
    estimates[i] = compute_expectation(n_sample.astype(int))
  return np.mean(estimates), np.var(estimates)

```

```python

# Compute the mean and variance for 1,2,... 20 samples
n_sample_all = np.array([1.,2,3,4,5,6,7,8,9,10,15,20,25,30,45,50,60,70,80,90,100,150,200,250,300,350,400,450,500])
mean_all = np.zeros_like(n_sample_all)
variance_all = np.zeros_like(n_sample_all)
for i in range(len(n_sample_all)):
  mean_all[i],variance_all[i] = compute_mean_variance(n_sample_all[i])
  print("No samples: ", n_sample_all[i], ", Mean: ", mean_all[i], ", Variance: ", variance_all[i])

```

```python

fig,ax = plt.subplots()
ax.semilogx(n_sample_all, mean_all,'r-',label='mean estimate')
ax.fill_between(n_sample_all, mean_all-2*np.sqrt(variance_all), mean_all+2*np.sqrt(variance_all))
ax.set_xlabel("Number of samples")
ax.set_ylabel("Mean of estimate")
ax.plot([0,500],[0.43160702267383166,0.43160702267383166],'k--',label='true value')
ax.legend()
plt.show()

```

As you might expect, the more samples that we use to compute the approximate estimate, the lower the variance of the estimate.

Now consider the function
 \begin{equation}
 \mbox{f}[y]= 20.446\exp\left[-(y-3)^4\right],
 \end{equation}

which decreases rapidly as we move away from the position $y=3$.

```python

def f2(y):
  return 20.446*np.exp(- (y-3) *(y-3) *(y-3) * (y-3))

fig,ax = plt.subplots()
y = np.arange(-10,10,0.01)
ax.plot(y, f2(y), 'r-', label='f$[y]$');
ax.plot(y, pr_y(y),'b-',label='$Pr(y)$')
ax.set_xlabel("$y$")
ax.legend()
plt.show()

```

Let's again, compute the expectation:

\begin{align}
\mathbb{E}_{y}\left[\text{f}[y]\right] &=& \int \text{f}[y] Pr(y) dy\\
&\approx& \frac{1}{I} \text{f}[y]
\end{align}

where $Pr(y)=\text{Norm}_y[0,1]$ by approximating with samples $y_{i}$.

```python

def compute_expectation2(n_samples):
  y = np.random.normal(size=(n_samples,1))
  expectation = np.mean(f2(y))

  return expectation

```

```python

# Set the seed so the random numbers are all the same
np.random.seed(0)

# Compute the expectation with a very large number of samples (good estimate)
n_samples = 100000000
expected_f2= compute_expectation2(n_samples)
print("Expected value: ", expected_f2)

```

I deliberately chose this function, because it's expectation is roughly the same as for the previous function.

Again, let's look at the mean and the  variance of the estimates

```python

def compute_mean_variance2(n_sample):
  n_estimate = 10000
  estimates = np.zeros((n_estimate,1))
  for i in range(n_estimate):
    estimates[i] = compute_expectation2(n_sample.astype(int))
  return np.mean(estimates), np.var(estimates)

# Compute the variance for 1,2,... 20 samples
mean_all2 = np.zeros_like(n_sample_all)
variance_all2 = np.zeros_like(n_sample_all)
for i in range(len(n_sample_all)):
  mean_all2[i], variance_all2[i] = compute_mean_variance2(n_sample_all[i])
  print("No samples: ", n_sample_all[i], ", Mean: ", mean_all2[i], ", Variance: ", variance_all2[i])

```

```python

fig,ax1 = plt.subplots()
ax1.semilogx(n_sample_all, mean_all,'r-',label='mean estimate')
ax1.fill_between(n_sample_all, mean_all-2*np.sqrt(variance_all), mean_all+2*np.sqrt(variance_all))
ax1.set_xlabel("Number of samples")
ax1.set_ylabel("Mean of estimate")
ax1.plot([1,500],[0.43160702267383166,0.43160702267383166],'k--',label='true value')
ax1.set_ylim(-5,6)
ax1.set_title("First function")
ax1.legend()

fig2,ax2 = plt.subplots()
ax2.semilogx(n_sample_all, mean_all2,'r-',label='mean estimate')
ax2.fill_between(n_sample_all, mean_all2-2*np.sqrt(variance_all2), mean_all2+2*np.sqrt(variance_all2))
ax2.set_xlabel("Number of samples")
ax2.set_ylabel("Mean of estimate")
ax2.plot([0,500],[0.43160428638892556,0.43160428638892556],'k--',label='true value')
ax2.set_ylim(-5,6)
ax2.set_title("Second function")
ax2.legend()
plt.show()

```

You can see that the variance of the estimate of the second function is considerably worse than the estimate of the variance of estimate of the first function

TODO:  Think about why this is.

Now let's repeat this experiment with the second function, but this time use importance sampling with auxiliary distribution:

 \begin{equation}
   q(y)=\text{Norm}_{y}[3,1]
 \end{equation}

```python

def q_y(y):
  return (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * (y-3) * (y-3))

def compute_expectation2b(n_samples):
  # TODO -- complete this function
  # 1. Draw n_samples from auxiliary distribution
  # 2. Compute f2[y] for those samples
  # 3. Scale the results by pr_y / q_y
  # 4. Compute the mean of these weighted samples
  # Replace this line
  expectation = 0

  return expectation

```

```python

# Set the seed so the random numbers are all the same
np.random.seed(0)

# Compute the expectation with a very large number of samples (good estimate)
n_samples = 100000000
expected_f2= compute_expectation2b(n_samples)
print("Your value: ", expected_f2,", True value:  0.43163734204459125 ")

```

```python

def compute_mean_variance2b(n_sample):
  n_estimate = 10000
  estimates = np.zeros((n_estimate,1))
  for i in range(n_estimate):
    estimates[i] = compute_expectation2b(n_sample.astype(int))
  return np.mean(estimates), np.var(estimates)

# Compute the variance for 1,2,... 20 samples
mean_all2b = np.zeros_like(n_sample_all)
variance_all2b = np.zeros_like(n_sample_all)
for i in range(len(n_sample_all)):
  mean_all2b[i], variance_all2b[i] = compute_mean_variance2b(n_sample_all[i])
  print("No samples: ", n_sample_all[i], ", Mean: ", mean_all2b[i], ", Variance: ", variance_all2b[i])

```

```python

fig,ax1 = plt.subplots()
ax1.semilogx(n_sample_all, mean_all,'r-',label='mean estimate')
ax1.fill_between(n_sample_all, mean_all-2*np.sqrt(variance_all), mean_all+2*np.sqrt(variance_all))
ax1.set_xlabel("Number of samples")
ax1.set_ylabel("Mean of estimate")
ax1.plot([1,500],[0.43160702267383166,0.43160702267383166],'k--',label='true value')
ax1.set_ylim(-5,6)
ax1.set_title("First function")
ax1.legend()

fig2,ax2 = plt.subplots()
ax2.semilogx(n_sample_all, mean_all2,'r-',label='mean estimate')
ax2.fill_between(n_sample_all, mean_all2-2*np.sqrt(variance_all2), mean_all2+2*np.sqrt(variance_all2))
ax2.set_xlabel("Number of samples")
ax2.set_ylabel("Mean of estimate")
ax2.plot([0,500],[0.43160428638892556,0.43160428638892556],'k--',label='true value')
ax2.set_ylim(-5,6)
ax2.set_title("Second function")
ax2.legend()
plt.show()

fig2,ax2 = plt.subplots()
ax2.semilogx(n_sample_all, mean_all2b,'r-',label='mean estimate')
ax2.fill_between(n_sample_all, mean_all2b-2*np.sqrt(variance_all2b), mean_all2b+2*np.sqrt(variance_all2b))
ax2.set_xlabel("Number of samples")
ax2.set_ylabel("Mean of estimate")
ax2.plot([0,500],[ 0.43163734204459125, 0.43163734204459125],'k--',label='true value')
ax2.set_ylim(-5,6)
ax2.set_title("Second function with importance sampling")
ax2.legend()
plt.show()

```

You can see that the importance sampling technique has reduced the amount of variance for any given number of samples.

***
