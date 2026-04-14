# Chap 9: 正则化 (Regularization)

> UDLbook 精读笔记
>
> **官方资源**: [GitHub Notebooks](https://github.com/udlbook/udlbook/tree/main/Notebooks/Chap09)

---

## Notebook 列表

- **L2 正则化**: `Chap09/9_1_L2_Regularization.ipynb`
- **隐式正则化**: `Chap09/9_2_Implicit_Regularization.ipynb`
- **集成学习**: `Chap09/9_3_Ensembling.ipynb`
- **贝叶斯方法**: `Chap09/9_4_Bayesian_Approach.ipynb`
- **数据增强**: `Chap09/9_5_Augmentation.ipynb`

---

## 内容

# **Notebook 9.1: L2 Regularization**

This notebook investigates adding L2 regularization to the loss function for the Gabor model as in figure 9.1.

Work through the cells below, running each cell in turn. In various places you will see the words "TODO". Follow the instructions at these places and make predictions about what is going to happen or write code to complete the functions.

Contact me at udlbookmail@gmail.com if you find any mistakes or have any suggestions.

```python

# import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

```

```python

# Let's create our training data 30 pairs {x_i, y_i}
# We'll try to fit the Gabor model to these data
data = np.array([[-1.920e+00,-1.422e+01,1.490e+00,-1.940e+00,-2.389e+00,-5.090e+00,
                 -8.861e+00,3.578e+00,-6.010e+00,-6.995e+00,3.634e+00,8.743e-01,
                 -1.096e+01,4.073e-01,-9.467e+00,8.560e+00,1.062e+01,-1.729e-01,
                  1.040e+01,-1.261e+01,1.574e-01,-1.304e+01,-2.156e+00,-1.210e+01,
                 -1.119e+01,2.902e+00,-8.220e+00,-1.179e+01,-8.391e+00,-4.505e+00],
                  [-1.051e+00,-2.482e-02,8.896e-01,-4.943e-01,-9.371e-01,4.306e-01,
                  9.577e-03,-7.944e-02 ,1.624e-01,-2.682e-01,-3.129e-01,8.303e-01,
                  -2.365e-02,5.098e-01,-2.777e-01,3.367e-01,1.927e-01,-2.222e-01,
                  6.352e-02,6.888e-03,3.224e-02,1.091e-02,-5.706e-01,-5.258e-02,
                  -3.666e-02,1.709e-01,-4.805e-02,2.008e-01,-1.904e-01,5.952e-01]])

```

```python

# Gabor model definition
def model(phi,x):
  sin_component = np.sin(phi[0] + 0.06 * phi[1] * x)
  gauss_component = np.exp(-(phi[0] + 0.06 * phi[1] * x) * (phi[0] + 0.06 * phi[1] * x) / 32)
  y_pred= sin_component * gauss_component
  return y_pred

```

```python

# Draw model
def draw_model(data,model,phi,title=None):
  x_model = np.arange(-15,15,0.1)
  y_model = model(phi,x_model)

  fix, ax = plt.subplots()
  ax.plot(data[0,:],data[1,:],'bo')
  ax.plot(x_model,y_model,'m-')
  ax.set_xlim([-15,15]);ax.set_ylim([-1,1])
  ax.set_xlabel('x'); ax.set_ylabel('y')
  if title is not None:
    ax.set_title(title)
  plt.show()

```

```python

# Initialize the parameters and draw the model
phi = np.zeros((2,1))
phi[0] =  -5     # Horizontal offset
phi[1] =  25     # Frequency
draw_model(data,model,phi, "Initial parameters")

```

Now let's
compute the sum of squares loss for the training data

```python

def compute_loss(data_x, data_y, model, phi):
  pred_y = model(phi, data_x)
  loss = np.sum((pred_y-data_y)*(pred_y-data_y))
  return loss

```

Now let's plot the whole loss function

```python

# Define pretty colormap
my_colormap_vals_hex =('2a0902', '2b0a03', '2c0b04', '2d0c05', '2e0c06', '2f0d07', '300d08', '310e09', '320f0a', '330f0b', '34100b', '35110c', '36110d', '37120e', '38120f', '39130f', '3a1410', '3b1411', '3c1511', '3d1612', '3e1613', '3f1713', '401714', '411814', '421915', '431915', '451a16', '461b16', '471b17', '481c17', '491d18', '4a1d18', '4b1e19', '4c1f19', '4d1f1a', '4e201b', '50211b', '51211c', '52221c', '53231d', '54231d', '55241e', '56251e', '57261f', '58261f', '592720', '5b2821', '5c2821', '5d2922', '5e2a22', '5f2b23', '602b23', '612c24', '622d25', '632e25', '652e26', '662f26', '673027', '683027', '693128', '6a3229', '6b3329', '6c342a', '6d342a', '6f352b', '70362c', '71372c', '72372d', '73382e', '74392e', '753a2f', '763a2f', '773b30', '783c31', '7a3d31', '7b3e32', '7c3e33', '7d3f33', '7e4034', '7f4134', '804235', '814236', '824336', '834437', '854538', '864638', '874739', '88473a', '89483a', '8a493b', '8b4a3c', '8c4b3c', '8d4c3d', '8e4c3e', '8f4d3f', '904e3f', '924f40', '935041', '945141', '955242', '965343', '975343', '985444', '995545', '9a5646', '9b5746', '9c5847', '9d5948', '9e5a49', '9f5a49', 'a05b4a', 'a15c4b', 'a35d4b', 'a45e4c', 'a55f4d', 'a6604e', 'a7614e', 'a8624f', 'a96350', 'aa6451', 'ab6552', 'ac6552', 'ad6653', 'ae6754', 'af6855', 'b06955', 'b16a56', 'b26b57', 'b36c58', 'b46d59', 'b56e59', 'b66f5a', 'b7705b', 'b8715c', 'b9725d', 'ba735d', 'bb745e', 'bc755f', 'bd7660', 'be7761', 'bf7862', 'c07962', 'c17a63', 'c27b64', 'c27c65', 'c37d66', 'c47e67', 'c57f68', 'c68068', 'c78169', 'c8826a', 'c9836b', 'ca846c', 'cb856d', 'cc866e', 'cd876f', 'ce886f', 'ce8970', 'cf8a71', 'd08b72', 'd18c73', 'd28d74', 'd38e75', 'd48f76', 'd59077', 'd59178', 'd69279', 'd7937a', 'd8957b', 'd9967b', 'da977c', 'da987d', 'db997e', 'dc9a7f', 'dd9b80', 'de9c81', 'de9d82', 'df9e83', 'e09f84', 'e1a185', 'e2a286', 'e2a387', 'e3a488', 'e4a589', 'e5a68a', 'e5a78b', 'e6a88c', 'e7aa8d', 'e7ab8e', 'e8ac8f', 'e9ad90', 'eaae91', 'eaaf92', 'ebb093', 'ecb295', 'ecb396', 'edb497', 'eeb598', 'eeb699', 'efb79a', 'efb99b', 'f0ba9c', 'f1bb9d', 'f1bc9e', 'f2bd9f', 'f2bfa1', 'f3c0a2', 'f3c1a3', 'f4c2a4', 'f5c3a5', 'f5c5a6', 'f6c6a7', 'f6c7a8', 'f7c8aa', 'f7c9ab', 'f8cbac', 'f8ccad', 'f8cdae', 'f9ceb0', 'f9d0b1', 'fad1b2', 'fad2b3', 'fbd3b4', 'fbd5b6', 'fbd6b7', 'fcd7b8', 'fcd8b9', 'fcdaba', 'fddbbc', 'fddcbd', 'fddebe', 'fddfbf', 'fee0c1', 'fee1c2', 'fee3c3', 'fee4c5', 'ffe5c6', 'ffe7c7', 'ffe8c9', 'ffe9ca', 'ffebcb', 'ffeccd', 'ffedce', 'ffefcf', 'fff0d1', 'fff2d2', 'fff3d3', 'fff4d5', 'fff6d6', 'fff7d8', 'fff8d9', 'fffada', 'fffbdc', 'fffcdd', 'fffedf', 'ffffe0')
my_colormap_vals_dec = np.array([int(element,base=16) for element in my_colormap_vals_hex])
r = np.floor(my_colormap_vals_dec/(256*256))
g = np.floor((my_colormap_vals_dec - r *256 *256)/256)
b = np.floor(my_colormap_vals_dec - r * 256 *256 - g * 256)
my_colormap = ListedColormap(np.vstack((r,g,b)).transpose()/255.0)

def draw_loss_function(compute_loss, data,  model, my_colormap, phi_iters = None):

  # Make grid of offset/frequency values to plot
  offsets_mesh, freqs_mesh = np.meshgrid(np.arange(-10,10.0,0.1), np.arange(2.5,22.5,0.1))
  loss_mesh = np.zeros_like(freqs_mesh)
  # Compute loss for every set of parameters
  for idslope, slope in np.ndenumerate(freqs_mesh):
     loss_mesh[idslope] = compute_loss(data[0,:], data[1,:], model, np.array([[offsets_mesh[idslope]], [slope]]))

  fig,ax = plt.subplots()
  fig.set_size_inches(8,8)
  ax.contourf(offsets_mesh,freqs_mesh,loss_mesh,256,cmap=my_colormap)
  ax.contour(offsets_mesh,freqs_mesh,loss_mesh,20,colors=['#80808080'])
  if phi_iters is not None:
    ax.plot(phi_iters[0,:], phi_iters[1,:],'go-')
  ax.set_ylim([2.5,22.5])
  ax.set_xlabel('Offset $\phi_{0}$'); ax.set_ylabel('Frequency, $\phi_{1}$')
  plt.show()

```

```python

draw_loss_function(compute_loss, data, model, my_colormap)

```

Now let's compute the gradient vector for a given set of parameters:

\begin{equation}
\frac{\partial L}{\partial \boldsymbol\phi} = \begin{bmatrix}\frac{\partial L}{\partial \phi_0} \\\frac{\partial L}{\partial \phi_1} \end{bmatrix}.
\end{equation}

```python

# These came from writing out the expression for the sum of squares loss and taking the
# derivative with respect to phi0 and phi1. It was a lot of hassle to get it right!
def gabor_deriv_phi0(data_x,data_y,phi0, phi1):
    x = 0.06 * phi1 * data_x + phi0
    y = data_y
    cos_component = np.cos(x)
    sin_component = np.sin(x)
    gauss_component = np.exp(-0.5 * x *x / 16)
    deriv = cos_component * gauss_component - sin_component * gauss_component * x / 16
    deriv = 2* deriv * (sin_component * gauss_component - y)
    return np.sum(deriv)

def gabor_deriv_phi1(data_x, data_y,phi0, phi1):
    x = 0.06 * phi1 * data_x + phi0
    y = data_y
    cos_component = np.cos(x)
    sin_component = np.sin(x)
    gauss_component = np.exp(-0.5 * x *x / 16)
    deriv = 0.06 * data_x * cos_component * gauss_component - 0.06 * data_x*sin_component * gauss_component * x / 16
    deriv = 2*deriv * (sin_component * gauss_component - y)
    return np.sum(deriv)

def compute_gradient(data_x, data_y, phi):
    dl_dphi0 = gabor_deriv_phi0(data_x, data_y, phi[0],phi[1])
    dl_dphi1 = gabor_deriv_phi1(data_x, data_y, phi[0],phi[1])
    # Return the gradient
    return np.array([[dl_dphi0],[dl_dphi1]])

```

Now we are ready to find the minimum.  For simplicity, we'll just use regular (non-stochastic) gradient descent with a fixed learning rate.

```python

def gradient_descent_step(phi, data,  model):
  # Step 1:  Compute the gradient
  gradient = compute_gradient(data[0,:],data[1,:], phi)
  # Step 2:  Update the parameters -- note we want to search in the negative (downhill direction)
  alpha = 0.1
  phi = phi - alpha * gradient
  return phi

```

```python

# Initialize the parameters
n_steps = 41
phi_all = np.zeros((2,n_steps+1))
phi_all[0,0] = 2.6
phi_all[1,0] = 8.5

# Measure loss and draw initial model
loss =  compute_loss(data[0,:], data[1,:], model, phi_all[:,0:1])
draw_model(data,model,phi_all[:,0:1], "Initial parameters, Loss = %f"%(loss))

for c_step in range (n_steps):
  # Do gradient descent step
  phi_all[:,c_step+1:c_step+2] = gradient_descent_step(phi_all[:,c_step:c_step+1],data, model)
  # Measure loss and draw model every 8th step
  if c_step % 8 == 0:
    loss =  compute_loss(data[0,:], data[1,:], model, phi_all[:,c_step+1:c_step+2])
    draw_model(data,model,phi_all[:,c_step+1], "Iteration %d, loss = %f"%(c_step+1,loss))

draw_loss_function(compute_loss, data, model, my_colormap, phi_all)

```

Unfortunately, when we start from this position, the solution descends to a local minimum and the final model doesn't fit well.<br><br>

But what if we had some weak knowledge that the solution was in the vicinity of $\phi_0=0.0$, $\phi_{1} = 12.5$ (the center of the plot)?

Let's add a term to the loss function that penalizes solutions that deviate from this point.  

\begin{equation}
L'[\boldsymbol\phi] = L[\boldsymbol\phi]+ \lambda\cdot \Bigl(\phi_{0}^2+(\phi_1-12.5)^2\Bigr)
\end{equation}

where $\lambda$ controls the relative importance of the original loss and the regularization term

```python

# Computes the regularization term
def compute_reg_term(phi0,phi1):
  # TODO compute the regularization term (term in large brackets in the above equation)
  # Replace this line
  reg_term = 0.0

  return reg_term ;

# Define the loss function
# Note I called the weighting lambda_ to avoid confusing it with python lambda functions
def compute_loss2(data_x, data_y, model, phi, lambda_):
  pred_y = model(phi, data_x)
  loss = np.sum((pred_y-data_y)*(pred_y-data_y))
  # Add the new term to the loss
  loss = loss + lambda_ * compute_reg_term(phi[0],phi[1])

  return loss

```

```python

# Code to draw the regularization function
def draw_reg_function():

  # Make grid of offset/frequency values to plot
  offsets_mesh, freqs_mesh = np.meshgrid(np.arange(-10,10.0,0.1), np.arange(2.5,22.5,0.1))
  loss_mesh = np.zeros_like(freqs_mesh)
  # Compute loss for every set of parameters
  for idslope, slope in np.ndenumerate(freqs_mesh):
     loss_mesh[idslope] = compute_reg_term(offsets_mesh[idslope], slope)

  fig,ax = plt.subplots()
  fig.set_size_inches(8,8)
  ax.contourf(offsets_mesh,freqs_mesh,loss_mesh,256,cmap=my_colormap)
  ax.contour(offsets_mesh,freqs_mesh,loss_mesh,20,colors=['#80808080'])
  ax.set_ylim([2.5,22.5])
  ax.set_xlabel('Offset $\phi_{0}$'); ax.set_ylabel('Frequency, $\phi_{1}$')
  plt.show()

# Draw the regularization function.  It should look similar to figure 9.1b
draw_reg_function()

```

```python

# Code to draw loss function with regularization
def draw_loss_function_reg(data,  model, lambda_, my_colormap, phi_iters = None):

  # Make grid of offset/frequency values to plot
  offsets_mesh, freqs_mesh = np.meshgrid(np.arange(-10,10.0,0.1), np.arange(2.5,22.5,0.1))
  loss_mesh = np.zeros_like(freqs_mesh)
  # Compute loss for every set of parameters
  for idslope, slope in np.ndenumerate(freqs_mesh):
     loss_mesh[idslope] = compute_loss2(data[0,:], data[1,:], model, np.array([[offsets_mesh[idslope]], [slope]]), lambda_)

  fig,ax = plt.subplots()
  fig.set_size_inches(8,8)
  ax.contourf(offsets_mesh,freqs_mesh,loss_mesh,256,cmap=my_colormap)
  ax.contour(offsets_mesh,freqs_mesh,loss_mesh,20,colors=['#80808080'])
  if phi_iters is not None:
    ax.plot(phi_iters[0,:], phi_iters[1,:],'go-')
  ax.set_ylim([2.5,22.5])
  ax.set_xlabel('Offset $\phi_{0}$'); ax.set_ylabel('Frequency, $\phi_{1}$')
  plt.show()

# This should look something like figure 9.1c
draw_loss_function_reg(data, model, 0.2, my_colormap)

```

```python

# TODO -- Experiment with different values of the regularization weight lambda_
# What do you predict will happen when it is very small (e.g. 0.01)?
# What do you predict will happen when it is large (e.g, 1.0)?
# What happens to the loss at the global minimum when we add the regularization term?
# Does it go up?  Go down?  Stay the same?

```

Now we'll compute the derivatives $\frac{\partial L'}{\partial\phi_0}$ and $\frac{\partial L'}{\partial\phi_1}$ of the regularized loss function:

\begin{equation}
L'[\boldsymbol\phi] = L[\boldsymbol\phi]+ \lambda\cdot \Bigl(\phi_{0}^2+(\phi_1-12.5)^2\Bigr)
\end{equation}

so that we can perform gradient descent.

```python

def dregdphi0(phi, lambda_):
  # TODO compute the derivative with respect to phi0
  # Replace this line:]
  deriv = 0

  return deriv

def dregdphi1(phi, lambda_):
  # TODO compute the derivative with respect to phi1
  # Replace this line:]
  deriv = 0


  return deriv


def compute_gradient2(data_x, data_y, phi, lambda_):
    dl_dphi0 = gabor_deriv_phi0(data_x, data_y, phi[0],phi[1])+dregdphi0(np.squeeze(phi), lambda_)
    dl_dphi1 = gabor_deriv_phi1(data_x, data_y, phi[0],phi[1])+dregdphi1(np.squeeze(phi), lambda_)
    # Return the gradient
    return np.array([[dl_dphi0],[dl_dphi1]])

def gradient_descent_step2(phi, lambda_, data,  model):
  # Step 1:  Compute the gradient
  gradient = compute_gradient2(data[0,:],data[1,:], phi, lambda_)
  # Step 2:  Update the parameters -- note we want to search in the negative (downhill direction)
  alpha = 0.1
  phi = phi - alpha * gradient
  return phi

```

```python

# Finally, let's run gradient descent and draw the result
# Initialize the parameters
n_steps = 41
phi_all = np.zeros((2,n_steps+1))
phi_all[0,0] = 2.6
phi_all[1,0] = 8.5
lambda_ = 0.2

# Measure loss and draw initial model
loss =  compute_loss2(data[0,:], data[1,:], model, phi_all[:,0:1], lambda_)
draw_model(data,model,phi_all[:,0:1], "Initial parameters, Loss = %f"%(loss))

for c_step in range (n_steps):
  # Do gradient descent step
  phi_all[:,c_step+1:c_step+2] = gradient_descent_step2(phi_all[:,c_step:c_step+1],lambda_, data, model)
  # Measure loss and draw model every 8th step
  if c_step % 8 == 0:
    loss =  compute_loss2(data[0,:], data[1,:], model, phi_all[:,c_step+1:c_step+2], lambda_)
    draw_model(data,model,phi_all[:,c_step+1], "Iteration %d, loss = %f"%(c_step+1,loss))

draw_loss_function_reg(data, model, lambda_, my_colormap, phi_all)

```

You should see that the gradient descent algorithm now finds the correct minimum.  By applying a tiny bit of domain knowledge (the parameter phi0 tends to be near zero and the parameter phi1 tends to be near 12.5), we get a better solution.  However, the cost is that this solution is slightly biased towards this prior knowledge.

***


# **Notebook 9.2: Implicit Regularization**

This notebook investigates how the finite step sizes in gradient descent cause the trajectory to deviate and how this can be explained by adding an implicit regularization term.  It recreates figure 9.3 from the book.

Work through the cells below, running each cell in turn. In various places you will see the words "TODO". Follow the instructions at these places and make predictions about what is going to happen or write code to complete the functions.

Contact me at udlbookmail@gmail.com if you find any mistakes or have any suggestions.

```python

# import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

```

```python

#Create colormap
my_colormap_vals_hex =('2a0902', '2b0a03', '2c0b04', '2d0c05', '2e0c06', '2f0d07', '300d08', '310e09', '320f0a', '330f0b', '34100b', '35110c', '36110d', '37120e', '38120f', '39130f', '3a1410', '3b1411', '3c1511', '3d1612', '3e1613', '3f1713', '401714', '411814', '421915', '431915', '451a16', '461b16', '471b17', '481c17', '491d18', '4a1d18', '4b1e19', '4c1f19', '4d1f1a', '4e201b', '50211b', '51211c', '52221c', '53231d', '54231d', '55241e', '56251e', '57261f', '58261f', '592720', '5b2821', '5c2821', '5d2922', '5e2a22', '5f2b23', '602b23', '612c24', '622d25', '632e25', '652e26', '662f26', '673027', '683027', '693128', '6a3229', '6b3329', '6c342a', '6d342a', '6f352b', '70362c', '71372c', '72372d', '73382e', '74392e', '753a2f', '763a2f', '773b30', '783c31', '7a3d31', '7b3e32', '7c3e33', '7d3f33', '7e4034', '7f4134', '804235', '814236', '824336', '834437', '854538', '864638', '874739', '88473a', '89483a', '8a493b', '8b4a3c', '8c4b3c', '8d4c3d', '8e4c3e', '8f4d3f', '904e3f', '924f40', '935041', '945141', '955242', '965343', '975343', '985444', '995545', '9a5646', '9b5746', '9c5847', '9d5948', '9e5a49', '9f5a49', 'a05b4a', 'a15c4b', 'a35d4b', 'a45e4c', 'a55f4d', 'a6604e', 'a7614e', 'a8624f', 'a96350', 'aa6451', 'ab6552', 'ac6552', 'ad6653', 'ae6754', 'af6855', 'b06955', 'b16a56', 'b26b57', 'b36c58', 'b46d59', 'b56e59', 'b66f5a', 'b7705b', 'b8715c', 'b9725d', 'ba735d', 'bb745e', 'bc755f', 'bd7660', 'be7761', 'bf7862', 'c07962', 'c17a63', 'c27b64', 'c27c65', 'c37d66', 'c47e67', 'c57f68', 'c68068', 'c78169', 'c8826a', 'c9836b', 'ca846c', 'cb856d', 'cc866e', 'cd876f', 'ce886f', 'ce8970', 'cf8a71', 'd08b72', 'd18c73', 'd28d74', 'd38e75', 'd48f76', 'd59077', 'd59178', 'd69279', 'd7937a', 'd8957b', 'd9967b', 'da977c', 'da987d', 'db997e', 'dc9a7f', 'dd9b80', 'de9c81', 'de9d82', 'df9e83', 'e09f84', 'e1a185', 'e2a286', 'e2a387', 'e3a488', 'e4a589', 'e5a68a', 'e5a78b', 'e6a88c', 'e7aa8d', 'e7ab8e', 'e8ac8f', 'e9ad90', 'eaae91', 'eaaf92', 'ebb093', 'ecb295', 'ecb396', 'edb497', 'eeb598', 'eeb699', 'efb79a', 'efb99b', 'f0ba9c', 'f1bb9d', 'f1bc9e', 'f2bd9f', 'f2bfa1', 'f3c0a2', 'f3c1a3', 'f4c2a4', 'f5c3a5', 'f5c5a6', 'f6c6a7', 'f6c7a8', 'f7c8aa', 'f7c9ab', 'f8cbac', 'f8ccad', 'f8cdae', 'f9ceb0', 'f9d0b1', 'fad1b2', 'fad2b3', 'fbd3b4', 'fbd5b6', 'fbd6b7', 'fcd7b8', 'fcd8b9', 'fcdaba', 'fddbbc', 'fddcbd', 'fddebe', 'fddfbf', 'fee0c1', 'fee1c2', 'fee3c3', 'fee4c5', 'ffe5c6', 'ffe7c7', 'ffe8c9', 'ffe9ca', 'ffebcb', 'ffeccd', 'ffedce', 'ffefcf', 'fff0d1', 'fff2d2', 'fff3d3', 'fff4d5', 'fff6d6', 'fff7d8', 'fff8d9', 'fffada', 'fffbdc', 'fffcdd', 'fffedf', 'ffffe0')
my_colormap_vals_dec = np.array([int(element,base=16) for element in my_colormap_vals_hex])
r = np.floor(my_colormap_vals_dec/(256*256))
g = np.floor((my_colormap_vals_dec - r *256 *256)/256)
b = np.floor(my_colormap_vals_dec - r * 256 *256 - g * 256)
my_colormap_vals = np.vstack((r,g,b)).transpose()/255.0
my_colormap = ListedColormap(my_colormap_vals)

```

```python

# define main function
def loss(phi0, phi1):
    phi1_std = np.exp(-0.5 * (phi0 * phi0)*4.0)
    return 1.0-np.exp(-0.5 * (phi1 * phi1)/(phi1_std * phi1_std))

# Compute the gradient (just done with finite differences for simplicity)
def get_loss_gradient(phi0, phi1):
    delta_phi = 0.00001;
    gradient = np.zeros((2,1));
    gradient[0] = (loss(phi0+delta_phi/2.0, phi1) - loss(phi0-delta_phi/2.0, phi1))/delta_phi
    gradient[1] = (loss(phi0, phi1+delta_phi/2.0) - loss(phi0, phi1-delta_phi/2.0))/delta_phi
    return gradient;

```

```python

# define grid to plot function
grid_values = np.arange(-0.8,0.5,0.01);
phi0mesh, phi1mesh = np.meshgrid(grid_values, grid_values)
loss_function = np.zeros((grid_values.size, grid_values.size))
for idphi0, phi0 in enumerate(grid_values):
    for idphi1, phi1 in enumerate(grid_values):
        loss_function[idphi0, idphi1] = loss(phi1,phi0)

```

```python

# Perform gradient descent n_steps times and return path
def grad_descent(start_posn, n_steps, step_size):
    grad_path = np.zeros((2, n_steps+1));
    grad_path[:,0] = start_posn[:,0];
    for c_step in range(n_steps):
        this_grad = get_loss_gradient(grad_path[0,c_step], grad_path[1,c_step]);
        grad_path[:,c_step+1] = grad_path[:,c_step] - step_size * this_grad[:,0]
    return grad_path;

```

```python

# Draw the loss function and the trajectories on it
def draw_function(phi0mesh, phi1mesh, loss_function, my_colormap, grad_path_tiny_lr=None, grad_path_typical_lr=None):
    fig = plt.figure();
    ax = plt.axes();
    fig.set_size_inches(7,7)
    ax.contourf(phi0mesh, phi1mesh, loss_function, 256, cmap=my_colormap);
    ax.contour(phi0mesh, phi1mesh, loss_function, 20, colors=['#80808080'])
    ax.set_xlabel(r'$\phi_{0}$'); ax.set_ylabel(r'$\phi_{1}$')

    if grad_path_typical_lr is not None:
        ax.plot(grad_path_typical_lr[0,:], grad_path_typical_lr[1,:],'ro-')
    if grad_path_tiny_lr is not None:
        ax.plot(grad_path_tiny_lr[0,:], grad_path_tiny_lr[1,:],'b-')
    plt.show()

```

```python

# Define the start position
start_posn = np.zeros((2,1)); start_posn[0,0] = -0.7; start_posn[1,0] = -0.75

# Run the gradient descent with a very small learning rate to simulate continuous case
grad_path_tiny_lr = grad_descent(start_posn, 10000, 0.001)
# Run the gradient descent with a typical sized learning rate
grad_path_typical_lr = grad_descent(start_posn, 100, 0.05)

draw_function(phi0mesh, phi1mesh, loss_function, my_colormap, grad_path_tiny_lr, grad_path_typical_lr)

```

You can see that the two solutions do not converge to the same place.  The ideal continuous solution is in blue, but in practice, we run the gradient set with as large a learning rate as possible so that it converges quickly (red curve). <br>

It turns out that using a large learning rate often gives better generalization results (figure 9.5a from book), and presumably, this is because we converge to a different (and better) place.

But how can we characterize the effect of the large learning rate?  One way is to consider what regularization term we would have to add to the original loss function so that the continuous solution converges to the same place as the discrete version with the large learning rate did on the original curve.

```python

# Compute the implicit regularization term (second term in equation 9.8 in the book)
def get_reg_term(phi0, phi1, alpha):
  # TODO -- compute this term
  # You can use the function get_loss_gradient(phi0, phi1) that was defined above
  # Replace this line:
  reg_term = 0.0;

  return reg_term;


# Compute modified loss function (equation 9.8)
def loss_reg(phi0, phi1, alpha):
    # The original function
    phi1_std = np.exp(-0.5 * (phi0 * phi0)*4.0)
    loss_out =  1.0-np.exp(-0.5 * (phi1 * phi1)/(phi1_std * phi1_std))

    # Add the regularization term that you just calculated above
    loss_out = loss_out + get_reg_term(phi0, phi1,alpha)
    return loss_out ;

# Compute gradient of modified loss function for gradient descent
def get_loss_gradient_reg(phi0, phi1,alpha):
    delta_phi = 0.00001;
    gradient = np.zeros((2,1));
    gradient[0] = (loss_reg(phi0+delta_phi/2.0, phi1, alpha) - loss_reg(phi0-delta_phi/2.0, phi1, alpha))/delta_phi
    gradient[1] = (loss_reg(phi0, phi1+delta_phi/2.0, alpha) - loss_reg(phi0, phi1-delta_phi/2.0, alpha))/delta_phi
    return gradient;

```

```python

# Let's visualize the regularization term
alpha = 0.1
reg_term = np.zeros((grid_values.size, grid_values.size))
for idphi0, phi0 in enumerate(grid_values):
    for idphi1, phi1 in enumerate(grid_values):
        reg_term[idphi0, idphi1] = get_reg_term(phi1,phi0, alpha)


draw_function(phi0mesh, phi1mesh, reg_term, my_colormap)

```

As you would expect, the regularization term is largest where the magnitude or the gradient of the original loss function was biggest (i.e., where it was steepest)

```python

# We'll also visualize the loss function plus the regularization term
alpha = 0.1
loss_function_reg = np.zeros((grid_values.size, grid_values.size))
for idphi0, phi0 in enumerate(grid_values):
    for idphi1, phi1 in enumerate(grid_values):
        loss_function_reg[idphi0, idphi1] = loss_reg (phi1,phi0, alpha)

draw_function(phi0mesh, phi1mesh, loss_function_reg, my_colormap)

```

It looks pretty similar to the original loss function, but you can see from the contours that it is slightly different.

```python

# Perform gradient descent n_steps times on modified loss function and return path
# Alpha is the step size for the gradient descent
# Alpha reg is the step size used to calculate the regularization term
def grad_descent_reg(start_posn, n_steps, alpha, alpha_reg):
    grad_path = np.zeros((2, n_steps+1));
    grad_path[:,0] = start_posn[:,0];
    for c_step in range(n_steps):
        this_grad = get_loss_gradient_reg(grad_path[0,c_step], grad_path[1,c_step],alpha_reg);
        grad_path[:,c_step+1] = grad_path[:,c_step] - alpha * this_grad[:,0]
    return grad_path;

```

```python

# Define the start position
start_posn = np.zeros((2,1)); start_posn[0,0] = -0.7; start_posn[1,0] = -0.75

# TODO:  Run the gradient descent on the modified loss
# function with 10000 steps and alpha_reg = 0.05, and a very small learning rate alpha of 0.001
# Replace this line:
grad_path_tiny_lr = None ;


# TODO:  Run the gradient descent on the unmodified loss
# function with 100 steps and a very small learning rate alpha of 0.05
# Replace this line:
grad_path_typical_lr = None ;


# Draw the functions
draw_function(phi0mesh, phi1mesh, loss_function_reg, my_colormap, grad_path_tiny_lr, grad_path_typical_lr)

```

Now the two trajectories align.  The red curve runs gradient descent with a typical step size on the original loss function.  The blue curve simulates continuous gradient descent on the regularized loss function.

***


# **Notebook 9.3: Ensembling**

This notebook investigates how ensembling can improve the performance of models. We'll work with the simplified neural network model (figure 8.4 of book) which we can fit in closed form, and so we can eliminate any errors due to not finding the global maximum.

Work through the cells below, running each cell in turn. In various places you will see the words "TODO". Follow the instructions at these places and make predictions about what is going to happen or write code to complete the functions.

Contact me at udlbookmail@gmail.com if you find any mistakes or have any suggestions.

```python

# import libraries
import numpy as np
import matplotlib.pyplot as plt
# Define seed to get same results each time
np.random.seed(1)

```

```python

# The true function that we are trying to estimate, defined on [0,1]
def true_function(x):
    y = np.exp(np.sin(x*(2*3.1413)))
    return y

```

```python

# Generate some data points with or without noise
def generate_data(n_data, sigma_y=0.3):
    # Generate x values quasi uniformly
    x = np.ones(n_data)
    for i in range(n_data):
        x[i] = np.random.uniform(i/n_data, (i+1)/n_data, 1)

    # y value from running through function and adding noise
    y = np.ones(n_data)
    for i in range(n_data):
        y[i] = true_function(x[i])
        y[i] += np.random.normal(0, sigma_y, 1)
    return x,y

```

```python

# Draw the fitted function, together with uncertainty used to generate points
def plot_function(x_func, y_func, x_data=None,y_data=None, x_model = None, y_model =None, sigma_func = None, sigma_model=None):

    fig,ax = plt.subplots()
    ax.plot(x_func, y_func, 'k-')
    if sigma_func is not None:
      ax.fill_between(x_func, y_func-2*sigma_func, y_func+2*sigma_func, color='lightgray')

    if x_data is not None:
        ax.plot(x_data, y_data, 'o', color='#d18362')

    if x_model is not None:
        ax.plot(x_model, y_model, '-', color='#7fe7de')

    if sigma_model is not None:
      ax.fill_between(x_model, y_model-2*sigma_model, y_model+2*sigma_model, color='lightgray')

    ax.set_xlim(0,1)
    ax.set_xlabel('Input, $x$')
    ax.set_ylabel('Output, $y$')
    plt.show()

```

```python

# Generate true function
x_func = np.linspace(0, 1.0, 100)
y_func = true_function(x_func);

# Generate some data points
np.random.seed(1)
sigma_func = 0.3
n_data = 15
x_data,y_data = generate_data(n_data, sigma_func)

# Plot the function, data and uncertainty
plot_function(x_func, y_func, x_data, y_data, sigma_func=sigma_func)

```

```python

# Define model -- beta is a scalar and omega has size n_hidden,1
def network(x, beta, omega):
    # Retrieve number of hidden units
    n_hidden = omega.shape[0]

    y = np.zeros_like(x)
    for c_hidden in range(n_hidden):
        # Evaluate activations based on shifted lines (figure 8.4b-d)
        line_vals =  x  - c_hidden/n_hidden
        h =  line_vals * (line_vals > 0)
        # Weight activations by omega parameters and sum
        y = y + omega[c_hidden] * h
    # Add bias, beta
    y = y + beta

    return y

```

```python

# This fits the n_hidden+1 parameters (see fig 8.4a) in closed form.
# If you have studied linear algebra, then you will know it is a least
# squares solution of the form (A^TA)^-1A^Tb.  If you don't recognize that,
# then just take it on trust that this gives you the best possible solution.
def fit_model_closed_form(x,y,n_hidden):
  n_data = len(x)
  A = np.ones((n_data, n_hidden+1))
  for i in range(n_data):
      for j in range(1,n_hidden+1):
          # Compute preactivation
          A[i,j] = x[i]-(j-1)/n_hidden
          # Apply the ReLU function
          if A[i,j] < 0:
              A[i,j] = 0;

  # Add a tiny bit of regularization
  reg_value = 0.00001
  regMat = reg_value * np.identity(n_hidden+1)
  regMat[0,0] = 0

  ATA = np.matmul(np.transpose(A), A) +regMat
  ATAInv = np.linalg.inv(ATA)
  ATAInvAT = np.matmul(ATAInv, np.transpose(A))
  beta_omega = np.matmul(ATAInvAT,y)
  beta = beta_omega[0]
  omega = beta_omega[1:]

  return beta, omega

```

```python

# Closed form solution
beta, omega = fit_model_closed_form(x_data,y_data,n_hidden=14)

# Get prediction for model across graph range
x_model = np.linspace(0,1,100);
y_model = network(x_model, beta, omega)

# Draw the function and the model
plot_function(x_func, y_func, x_data,y_data, x_model, y_model)

# Compute the mean squared error between the fitted model (cyan) and the true curve (black)
mean_sq_error = np.mean((y_model-y_func) * (y_model-y_func))
print(f"Mean square error = {mean_sq_error:3.3f}")

```

```python

# Now let's resample the data with replacement four times.
n_model = 4
# Array to store the prediction from all of our models
all_y_model = np.zeros((n_model, len(y_model)))

# For each model
for c_model in range(n_model):
    # TODO Sample data indices with replacement (use np.random.choice)
    # Replace this line
    resampled_indices = np.arange(0,n_data,1);

    # Extract the resampled x and y data
    x_data_resampled = x_data[resampled_indices]
    y_data_resampled = y_data[resampled_indices]

    # Fit the model
    beta, omega = fit_model_closed_form(x_data_resampled,y_data_resampled,n_hidden=14)

    # Run the model
    y_model_resampled = network(x_model, beta, omega)

    # Store the results
    all_y_model[c_model,:] = y_model_resampled

    # Draw the function and the model
    plot_function(x_func, y_func, x_data,y_data, x_model, y_model_resampled)

    # Compute the mean squared error between the fitted model (cyan) and the true curve (black)
    mean_sq_error = np.mean((y_model_resampled-y_func) * (y_model_resampled-y_func))
    print(f"Mean square error = {mean_sq_error:3.3f}")

```

```python

# Plot the median of the results
# TODO -- find the median prediction
# Replace this line
y_model_median = all_y_model[0,:]

# Draw the function and the model
plot_function(x_func, y_func, x_data,y_data, x_model, y_model_median)

# Compute the mean squared error between the fitted model (cyan) and the true curve (black)
mean_sq_error = np.mean((y_model_median-y_func) * (y_model_median-y_func))
print(f"Mean square error = {mean_sq_error:3.3f}")

```

```python

# Plot the mean of the results
# TODO -- find the mean prediction
# Replace this line
y_model_mean = all_y_model[0,:]

# Draw the function and the model
plot_function(x_func, y_func, x_data,y_data, x_model, y_model_mean)

# Compute the mean squared error between the fitted model (cyan) and the true curve (black)
mean_sq_error = np.mean((y_model_mean-y_func) * (y_model_mean-y_func))
print(f"Mean square error = {mean_sq_error:3.3f}")

```

You should see that both the median and mean models are better than any of the individual models. We have improved our performance at the cost of four times as much training time, storage, and inference time.

***


# **Notebook 9.4: Bayesian approach**

This notebook investigates the Bayesian approach to model fitting and reproduces figure 9.11 from the book.

Work through the cells below, running each cell in turn. In various places you will see the words "TODO". Follow the instructions at these places and make predictions about what is going to happen or write code to complete the functions.

Contact me at udlbookmail@gmail.com if you find any mistakes or have any suggestions.

```python

# import libraries
import numpy as np
import matplotlib.pyplot as plt
# Define seed to get same results each time
np.random.seed(1)

```

```python

# The true function that we are trying to estimate, defined on [0,1]
def true_function(x):
    y = np.exp(np.sin(x*(2*3.1413)))
    return y

```

```python

# Generate some data points with or without noise
def generate_data(n_data, sigma_y=0.3):
    # Generate x values quasi uniformly
    x = np.ones(n_data)
    for i in range(n_data):
        x[i] = np.random.uniform(i/n_data, (i+1)/n_data, 1)

    # y value from running through function and adding noise
    y = np.ones(n_data)
    for i in range(n_data):
        y[i] = true_function(x[i])
        y[i] += np.random.normal(0, sigma_y, 1)
    return x,y

```

```python

# Draw the fitted function, together with uncertainty used to generate points
def plot_function(x_func, y_func, x_data=None,y_data=None, x_model = None, y_model =None, sigma_func = None, sigma_model=None):

    fig,ax = plt.subplots()
    ax.plot(x_func, y_func, 'k-')
    if sigma_func is not None:
      ax.fill_between(x_func, y_func-2*sigma_func, y_func+2*sigma_func, color='lightgray')

    if x_data is not None:
        ax.plot(x_data, y_data, 'o', color='#d18362')

    if x_model is not None:
        ax.plot(x_model, y_model, '-', color='#7fe7de')

    if sigma_model is not None:
      ax.fill_between(x_model, y_model-2*sigma_model, y_model+2*sigma_model, color='lightgray')

    ax.set_xlim(0,1)
    ax.set_xlabel('Input, $x$')
    ax.set_ylabel('Output, $y$')
    plt.show()

```

```python

# Generate true function
x_func = np.linspace(0, 1.0, 100)
y_func = true_function(x_func);

# Generate some data points
np.random.seed(1)
sigma_func = 0.3
n_data = 15
x_data,y_data = generate_data(n_data, sigma_func)

# Plot the function, data and uncertainty
plot_function(x_func, y_func, x_data, y_data, sigma_func=sigma_func)

```

```python

# Define model -- beta is a scalar and omega has size n_hidden,1
def network(x, beta, omega):
    # Retrieve number of hidden units
    n_hidden = omega.shape[0]

    y = np.zeros_like(x)
    for c_hidden in range(n_hidden):
        # Evaluate activations based on shifted lines (figure 8.4b-d)
        line_vals =  x  - c_hidden/n_hidden
        h =  line_vals * (line_vals > 0)
        # Weight activations by omega parameters and sum
        y = y + omega[c_hidden] * h
    # Add bias, beta
    y = y + beta

    return y

```

Now let's compute a probability distribution over the model parameters using Bayes's rule:

\begin{equation}
 Pr(\boldsymbol\phi|\{\mathbf{x}_{i},\mathbf{y}_{i}\}) = \frac{\prod_{i=1}^{I} Pr(\mathbf{y}_{i}|\mathbf{x}_{i},\boldsymbol\phi) Pr(\boldsymbol\phi)}{\int \prod_{i=1}^{I} Pr(\mathbf{y}_{i}|\mathbf{x}_{i},\boldsymbol\phi) Pr(\boldsymbol\phi)d\boldsymbol\phi } ,
\end{equation}

We'll define the prior $Pr(\boldsymbol\phi)$ as:

\begin{equation}
Pr(\boldsymbol\phi) = \text{Norm}_{\boldsymbol\phi}\bigl[\mathbf{0},\sigma^2_p\mathbf{I}\bigr]
\end{equation}

where $\phi=[\omega_1,\omega_2\ldots \omega_n, \beta]^T$ and $\sigma^2_{p}$  is the prior variance.

The likelihood term $\prod_{i=1}^{I} Pr(\mathbf{y}_{i}|\mathbf{x}_{i},\boldsymbol\phi)$ is given by:

\begin{align}
\prod_{i=1}^{I} Pr(\mathbf{y}_{i}|\mathbf{x}_{i},\boldsymbol\phi) &=& \prod_{i=1}^{I} \text{Norm}_{y_i}\bigl[\text{f}[\mathbf{x}_{i},\boldsymbol\phi],\sigma_d^2\bigr]\\
&=& \prod_{i=1}^{I} \text{Norm}_{y_i}\bigl[\boldsymbol\omega\mathbf{h}_i+\beta,\sigma_d^2\bigr]\\
&=& \text{Norm}_{\mathbf{y}}\bigl[\mathbf{H}^T\boldsymbol\phi,\sigma^2\mathbf{I}\bigr].
\end{align}

where $\sigma^2$ is the measurement noise and $\mathbf{h}_{i}$ is the column vector of hidden variables for the $i^{th}$ input.  Here the vector $\mathbf{y}$ and matrix $\mathbf{H}$ are defined as:

\begin{equation}
\mathbf{y} = \begin{bmatrix}y_1\\y_2\\\vdots\\y_{I}\end{bmatrix}\quad\text{and}\quad \mathbf{H} = \begin{bmatrix}\mathbf{h}_{1}&\mathbf{h}_{2}&\cdots&\mathbf{h}_{I}\\1&1&\cdots &1\end{bmatrix}.
\end{equation}

To make progress we use the change of variable relation (Appendix C.3.4 of the book) to rewrite the likelihood term as a normal distribution in the parameters $\boldsymbol\phi$:

\begin{align}
\prod_{i=1}^{I} Pr(\mathbf{y}_{i}|\mathbf{x}_{i},\boldsymbol\phi+\beta)
&=& \text{Norm}_{\mathbf{y}}\bigl[\mathbf{H}^T\boldsymbol\phi,\sigma^2\bigr]\\
&\propto& \text{Norm}_{\boldsymbol\phi}\bigl[(\mathbf{H}\mathbf{H}^T)^{-1}\mathbf{H}\mathbf{y},\sigma^2(\mathbf{H}\mathbf{H}^t)^{-1}\bigr]
\end{align}

Finally, we can combine the likelihood and prior terms using the product of two normal distributions relation (Appendix C.3.3).

\begin{align}
 Pr(\boldsymbol\phi|\{\mathbf{x}_{i},\mathbf{y}_{i}\}) &\propto& \prod_{i=1}^{I} Pr(\mathbf{y}_{i}|\mathbf{x}_{i},\boldsymbol\phi) Pr(\boldsymbol\phi)\\
 &\propto&\text{Norm}_{\boldsymbol\phi}\bigl[(\mathbf{H}\mathbf{H}^T)^{-1}\mathbf{H}\mathbf{y},\sigma^2(\mathbf{H}\mathbf{H}^T)^{-1}\bigr] \text{Norm}_{\boldsymbol\phi}\bigl[\mathbf{0},\sigma^2_p\mathbf{I}\bigr]\\
 &\propto&\text{Norm}_{\boldsymbol\phi}\biggl[\frac{1}{\sigma^2}\left(\frac{1}{\sigma^2}\mathbf{H}\mathbf{H}^T+\frac{1}{\sigma_p^2}\mathbf{I}\right)^{-1}\mathbf{H}\mathbf{y},\left(\frac{1}{\sigma^2}\mathbf{H}\mathbf{H}^T+\frac{1}{\sigma_p^2}\mathbf{I}\right)^{-1}\biggr].
\end{align}

In fact, since this is already a normal distribution, the constant of proportionality must be one and we can write

\begin{align}
 Pr(\boldsymbol\phi|\{\mathbf{x}_{i},\mathbf{y}_{i}\}) &=& \text{Norm}_{\boldsymbol\phi}\biggl[\frac{1}{\sigma^2}\left(\frac{1}{\sigma^2}\mathbf{H}\mathbf{H}^T+\frac{1}{\sigma_p^2}\mathbf{I}\right)^{-1}\mathbf{H}\mathbf{y},\left(\frac{1}{\sigma^2}\mathbf{H}\mathbf{H}^T+\frac{1}{\sigma_p^2}\mathbf{I}\right)^{-1}\biggr].
\end{align}

TODO -- On a piece of paper, use the relations in Appendix C.3.3 and C.3.4 to fill in the missing steps and establish that this is the correct formula for the posterior.

```python

def compute_H(x_data, n_hidden):
  psi1 = np.ones((n_hidden+1,1));
  psi0 = np.linspace(0.0, 1.0, num=n_hidden, endpoint=False) * -1

  n_data = x_data.size
  # First compute the hidden variables
  H = np.ones((n_hidden+1, n_data))
  for i in range(n_hidden):
    for j in range(n_data):
      # Compute preactivation
      H[i,j] = psi1[i] * x_data[j]+psi0[i]
      # Apply ReLU to get activation
      if H[i,j] < 0:
        H[i,j] = 0;

  return H

def compute_param_mean_covar(x_data, y_data, n_hidden, sigma_sq, sigma_p_sq):
  # Retrieve the matrix containing the hidden variables
  H = compute_H(x_data, n_hidden) ;

  # TODO -- Compute the covariance matrix (you will need np.transpose(), np.matmul(), np.linalg.inv())
  # Replace this line
  phi_covar = np.identity(n_hidden+1)


  # TODO -- Compute the mean matrix
  # Replace this line
  phi_mean = np.zeros((n_hidden+1,1))


  return phi_mean, phi_covar

```

Now we can draw samples from this distribution

```python

# Define parameters
n_hidden = 5
sigma_sq = sigma_func * sigma_func
# Arbitrary large value reflecting the fact we are uncertain about the
# parameters before we see any data
sigma_p_sq = 1000

# Compute the mean and covariance matrix
phi_mean, phi_covar = compute_param_mean_covar(x_data, y_data, n_hidden, sigma_sq, sigma_p_sq)

# Let's draw the mean model
x_model = x_func
y_model_mean = network(x_model, phi_mean[-1], phi_mean[0:n_hidden])
plot_function(x_func, y_func, x_data, y_data, x_model, y_model_mean)

```

```python

# TODO Draw two samples from the normal distribution over the parameters
# Replace these lines
phi_sample1 = np.zeros((n_hidden+1,1))
phi_sample2 = np.zeros((n_hidden+1,1))


# Run the network for these two sample sets of parameters
y_model_sample1 = network(x_model, phi_sample1[-1], phi_sample1[0:n_hidden])
y_model_sample2 = network(x_model, phi_sample2[-1], phi_sample2[0:n_hidden])

# Draw the two models
plot_function(x_func, y_func, x_data, y_data, x_model, y_model_sample1)
plot_function(x_func, y_func, x_data, y_data, x_model, y_model_sample2)

```

Now we need to perform inference for a new data points $\mathbf{x}^*$ with corresponding hidden values $\mathbf{h}^*$.  Instead of having a single estimate of the parameters, we have a distribution over the possible parameters.  So we marginalize (integrate) over this distribution to account for all possible values:

\begin{align}
Pr(y^*|\mathbf{x}^*)  &= \int Pr(y^{*}|\mathbf{x}^*,\boldsymbol\phi)Pr(\boldsymbol\phi|\{\mathbf{x}_{i},\mathbf{y}_{i}\}) d\boldsymbol\phi\\
&= \int \text{Norm}_{y^*}\bigl[[\mathbf{h}^{*T},1]\boldsymbol\phi,\sigma^2\bigr]\cdot\text{Norm}_{\boldsymbol\phi}\biggl[\frac{1}{\sigma^2}\left(\frac{1}{\sigma^2}\mathbf{H}\mathbf{H}^T+\frac{1}{\sigma_p^2}\mathbf{I}\right)^{-1}\mathbf{H}\mathbf{y},\left(\frac{1}{\sigma^2}\mathbf{H}\mathbf{H}^T+\frac{1}{\sigma_p^2}\mathbf{I}\right)^{-1}\biggr]d\boldsymbol\phi\\
&= \text{Norm}_{y^*}\biggl[\frac{1}{\sigma^2} [\mathbf{h}^{*T},1]\left(\frac{1}{\sigma^2}\mathbf{H}\mathbf{H}^T+\frac{1}{\sigma_p^2}\mathbf{I}\right)^{-1}\mathbf{H}\mathbf{y},  [\mathbf{h}^{*T},1]\left(\frac{1}{\sigma^2}\mathbf{H}\mathbf{H}^T+\frac{1}{\sigma_p^2}\mathbf{I}\right)^{-1}
[\mathbf{h}^*;1]\biggr],
\end{align}

where the notation $[\mathbf{h}^{*T},1]$ is a row vector containing $\mathbf{h}^{*T}$ with a one appended to the end and $[\mathbf{h}^{*};1 ]$ is a column vector containing $\mathbf{h}^{*}$ with a one appended to the end.


To compute this, we reformulated the integrand using the relations from appendices C.3.3 and C.3.4 as the product of a normal distribution in $\boldsymbol\phi$ and a constant with respect
to $\boldsymbol\phi$. The integral of the normal distribution must be one, and so the final result is just the constant. This constant is itself a normal distribution in $y^*$. <br>

If you feel so inclined you can work through the math of this yourself.

```python

# Predict mean and variance of y_star from x_star
def inference(x_star, x_data, y_data, sigma_sq, sigma_p_sq, n_hidden):

  # Compute hidden variables
  h_star = compute_H(x_star, n_hidden);
  H = compute_H(x_data, n_hidden);

  # TODO: Compute mean and variance of y*
  # Replace these lines:
  y_star_mean = 0
  y_star_var =  1

  return y_star_mean, y_star_var

```

```python

x_model = x_func
y_model = np.zeros_like(x_model)
y_model_std = np.zeros_like(x_model)
for c_model in range(len(x_model)):
  y_star_mean, y_star_var = inference(x_model[c_model]*np.ones((1,1)), x_data, y_data, sigma_sq, sigma_p_sq, n_hidden)
  y_model[c_model] = y_star_mean
  y_model_std[c_model] = np.sqrt(y_star_var)

# Draw the model
plot_function(x_func, y_func, x_data, y_data, x_model, y_model, sigma_model=y_model_std)

```

TODO:

1.  Experiment running this again with different numbers of hidden units.  Make a prediction for what will happen when you increase / decrease them.
2.  Experiment with what happens if you make the prior variance $\sigma^2_p$ to a smaller value like 1.  How do you explain the results?

***


# **Notebook 9.5: Augmentation**

This notebook investigates data augmentation for the MNIST-1D model.

Work through the cells below, running each cell in turn. In various places you will see the words "TODO". Follow the instructions at these places and make predictions about what is going to happen or write code to complete the functions.

Contact me at udlbookmail@gmail.com if you find any mistakes or have any suggestions.

```python

# Run this if you're in a Colab to install MNIST 1D repository
!pip install git+https://github.com/greydanus/mnist1d

```

```python

import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
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

D_i = 40    # Input dimensions
D_k = 200   # Hidden dimensions
D_o = 10    # Output dimensions

# Define a model with two hidden layers of size 200
# And ReLU activations between them
model = nn.Sequential(
nn.Linear(D_i, D_k),
nn.ReLU(),
nn.Linear(D_k, D_k),
nn.ReLU(),
nn.Linear(D_k, D_o))

def weights_init(layer_in):
  # Initialize the parameters with He initialization
  if isinstance(layer_in, nn.Linear):
    nn.init.kaiming_uniform_(layer_in.weight)
    layer_in.bias.data.fill_(0.0)

```

```python

# choose cross entropy loss function (equation 5.24)
loss_function = torch.nn.CrossEntropyLoss()
# construct SGD optimizer and initialize learning rate and momentum
optimizer = torch.optim.SGD(model.parameters(), lr = 0.05, momentum=0.9)
# object that decreases learning rate by half every 10 epochs
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
# create 100 dummy data points and store in data loader class
x_train = torch.tensor(data['x'].astype('float32'))
y_train = torch.tensor(data['y'].transpose().astype('long'))
x_test= torch.tensor(data['x_test'].astype('float32'))
y_test = torch.tensor(data['y_test'].astype('long'))

# load the data into a class that creates the batches
data_loader = DataLoader(TensorDataset(x_train,y_train), batch_size=100, shuffle=True, worker_init_fn=np.random.seed(1))

# Initialize model weights
model.apply(weights_init)

# loop over the dataset n_epoch times
n_epoch = 50
# store the loss and the % correct at each epoch
errors_train = np.zeros((n_epoch))
errors_test = np.zeros((n_epoch))

for epoch in range(n_epoch):
  # loop over batches
  for i, batch in enumerate(data_loader):
    # retrieve inputs and labels for this batch
    x_batch, y_batch = batch
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
  pred_test = model(x_test)
  _, predicted_train_class = torch.max(pred_train.data, 1)
  _, predicted_test_class = torch.max(pred_test.data, 1)
  errors_train[epoch] = 100 - 100 * (predicted_train_class == y_train).float().sum() / len(y_train)
  errors_test[epoch]= 100 - 100 * (predicted_test_class == y_test).float().sum() / len(y_test)
  print(f'Epoch {epoch:5d}, train error {errors_train[epoch]:3.2f}, test error {errors_test[epoch]:3.2f}')

```

```python

# Plot the results
fig, ax = plt.subplots()
ax.plot(errors_train,'r-',label='train')
ax.plot(errors_test,'b-',label='test')
ax.set_ylim(0,100); ax.set_xlim(0,n_epoch)
ax.set_xlabel('Epoch'); ax.set_ylabel('Error')
ax.set_title('Train Error %3.2f, Test Error %3.2f'%(errors_train[-1],errors_test[-1]))
ax.legend()
plt.show()

```

The best test performance is about 33%.  Let's see if we can improve on that by augmenting the data.

```python

def augment(input_vector):
  # Create output vector
  data_out = np.zeros_like(input_vector)

  # TODO:  Shift the input data by a random offset
  # (rotating, so points that would go off the end, are added back to the beginning)
  # Replace this line:
  data_out = np.zeros_like(input_vector) ;

  # TODO:    # Randomly scale the data by a factor drawn from a uniform distribution over [0.8,1.2]
  # Replace this line:
  data_out = np.array(data_out)

  return data_out

```

```python

n_data_orig = data['x'].shape[0]
# We'll double the amount of data
n_data_augment = n_data_orig+4000
augmented_x = np.zeros((n_data_augment, D_i))
augmented_y = np.zeros(n_data_augment)
# First n_data_orig rows are original data
augmented_x[0:n_data_orig,:] = data['x']
augmented_y[0:n_data_orig] = data['y']

# Fill in rest of with augmented data
for c_augment in range(n_data_orig, n_data_augment):
  # Choose a data point randomly
  random_data_index = random.randint(0, n_data_orig-1)
  # Augment the point and store
  augmented_x[c_augment,:] = augment(data['x'][random_data_index,:])
  augmented_y[c_augment] = data['y'][random_data_index]

```

```python

# choose cross entropy loss function (equation 5.24)
loss_function = torch.nn.CrossEntropyLoss()
# construct SGD optimizer and initialize learning rate and momentum
optimizer = torch.optim.SGD(model.parameters(), lr = 0.05, momentum=0.9)
# object that decreases learning rate by half every 50 epochs
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
# create 100 dummy data points and store in data loader class
x_train = torch.tensor(augmented_x.astype('float32'))
y_train = torch.tensor(augmented_y.transpose().astype('long'))
x_test= torch.tensor(data['x_test'].astype('float32'))
y_test = torch.tensor(data['y_test'].astype('long'))

# load the data into a class that creates the batches
data_loader = DataLoader(TensorDataset(x_train,y_train), batch_size=100, shuffle=True, worker_init_fn=np.random.seed(1))

# Initialize model weights
model.apply(weights_init)

# loop over the dataset n_epoch times
n_epoch = 50
# store the loss and the % correct at each epoch
errors_train_aug = np.zeros((n_epoch))
errors_test_aug = np.zeros((n_epoch))

for epoch in range(n_epoch):
  # loop over batches
  for i, batch in enumerate(data_loader):
    # retrieve inputs and labels for this batch
    x_batch, y_batch = batch
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
  pred_test = model(x_test)
  _, predicted_train_class = torch.max(pred_train.data, 1)
  _, predicted_test_class = torch.max(pred_test.data, 1)
  errors_train_aug[epoch] = 100 - 100 * (predicted_train_class == y_train).float().sum() / len(y_train)
  errors_test_aug[epoch]= 100 - 100 * (predicted_test_class == y_test).float().sum() / len(y_test)
  print(f'Epoch {epoch:5d}, train error {errors_train_aug[epoch]:3.2f}, test error {errors_test_aug[epoch]:3.2f}')

```

```python

# Plot the results
fig, ax = plt.subplots()
ax.plot(errors_train,'r-',label='train')
ax.plot(errors_test,'b-',label='test')
ax.plot(errors_test_aug,'g-',label='test (augmented)')
ax.set_ylim(0,100); ax.set_xlim(0,n_epoch)
ax.set_xlabel('Epoch'); ax.set_ylabel('Error')
ax.set_title('TrainError %3.2f, Test Error %3.2f'%(errors_train_aug[-1],errors_test_aug[-1]))
ax.legend()
plt.show()

```

Hopefully, you should see an improvement in performance when we augment the data.

***
