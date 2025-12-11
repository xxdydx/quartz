## tensors
### introduction

Tensors are the fundamental data structure. It is a specific data structure optimised for two things:
1. **GPU Acceleration:** Unlike NumPy arrays which live on the CPU, tensors can be moved to the GPU (CUDA) for massive parallelism.
2. **Automatic Differentiation:** Tensors have a built-in memory of the mathematical operations performed on them, allowing for automatic gradient calculation.

### creating and inspecting

In production, bugs usually stem from **Shape**, **DataType**, or **Device** mismatches.

```python
import torch

# Standard creation: Random tensor with specific shape
# Shape: [Channels, Height, Width]
x = torch.rand(size=(3, 224, 224)) 

# The "Big Three" attributes you must check
print(f"Shape: {x.shape}")        # torch.Size([3, 224, 224])
print(f"Datatype: {x.dtype}")     # torch.float32 (Standard for ML)
print(f"Device: {x.device}")      # cpu (Default)
```


### device agnostic code

Hardcoding `.cuda()` is a bad practice because it crashes on non-NVIDIA hardware (like MacBooks or CPU servers). This is how it should be done:
```python
# Standard production setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Moving a tensor to the accelerator
tensor_on_gpu = tensor_a.to(device)

# Note: Operations between CPU and GPU tensors will throw a Runtime Error.
# You must move them to the same device first.
```


### operations and broadcasting

```python
# Element-wise multiplication
z = x * x 

# Matrix Multiplication (The engine of Neural Nets)
# Rule: Inner dims must match (3, 2) @ (2, 5) -> (3, 5)
a = torch.randn(3, 2)
b = torch.randn(2, 5)
result = a @ b  # @ is the shortcut for torch.matmul(a, b)
```

PyTorch automatically "broadcasts" (stretches) smaller tensors to match larger ones during math operations.

**rules for broadcasting tensors**
1. Align shapes from the **right**.
2. A dimension is compatible if:
    - They are equal, OR
    - One of them is **1**.

```python
import torch

# Scenario: We have a batch of 2 data samples.
# Each sample has 3 features (e.g., x, y, z coordinates).
data = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
], dtype=torch.float32) 

print(f"Data Shape: {data.shape}") # torch.Size([2, 3])

# We want to add a specific bias to each feature.
# Bias for feature 1 is +10, feature 2 is +20, feature 3 is +30.
bias = torch.tensor([10, 20, 30], dtype=torch.float32)

print(f"Bias Shape: {bias.shape}") # torch.Size([3])

# THE BROADCASTING MAGIC
# PyTorch aligns:     [2, 3]
#                     [   3]
# Prepends 1:         [1, 3]
# Stretches 1 -> 2:   [2, 3] (Copies the bias row to match batch size)
result = data + bias 

print(result)
# Output:
# tensor([[11., 22., 33.],  <-- [1, 2, 3] + [10, 20, 30]
#         [14., 25., 36.]]) <-- [4, 5, 6] + [10, 20, 30] (Bias reused!)
```



## automatic differentiation

### the engine

PyTorch builds a **Directed Acyclic Graph (DAG)** of operations. Leaves are input tensors; roots are output/loss functions.
- **`requires_grad=True`**: Tells PyTorch to track every operation performed on this variable.
- **`grad_fn`**: The backward function stored in the tensor (e.g., `AddBackward`, `MulBackward`).
### the backward pass

```python
# 1. Create tensors with history tracking
w = torch.tensor([1.0], requires_grad=True)
x = torch.tensor([2.0]) 
b = torch.tensor([3.0], requires_grad=True)

# 2. Forward pass (Builds the graph)
y = w * x + b   # y = 1*2 + 3 = 5

# 3. Compute loss
loss = (y - 10)**2 

# 4. Backward pass (Calculates gradients)
loss.backward()

# 5. Inspect gradients (dL/dw)
# dL/dw = 2(y-10) * x = 2(-5)*2 = -20
print(w.grad)
```

### gradient accumulation

Gradients are **additive**. Calling `.backward()` multiple times adds to the `.grad` attribute rather than overwriting it. This is why `optimizer.zero_grad()` is mandatory in every training loop step.


## linear models & mlp

Every network inherits from `nn.Module`. You define layers in `__init__` and connectivity in `forward`.

### code: multi-layer perceptron (mlp)

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # nn.Sequential wraps layers into a single block
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),                   # Non-linearity
            nn.Dropout(p=0.2),           # Regularization
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)
```

### the training loop
```python
model = MLP(10, 64, 1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.MSELoss()

# 1. Forward Pass
preds = model(inputs)

# 2. Calculate Loss
loss = criterion(preds, targets)

# 3. Zero Gradients (Flush history)
optimizer.zero_grad()

# 4. Backward Pass (Compute grads)
loss.backward()

# 5. Optimizer Step (Update weights)
optimizer.step()
```


## working with images & convolution

### cnns: dealing with spatial data

Linear layers destroy spatial structure (pixels nearby matter). **Convolutions** preserve it.
### key components

1. **`nn.Conv2d`**: Learns local features (edges, textures).
    - _Kernel_: Size of the window (e.g., 3x3).
    - _Stride_: Step size.
    - _Padding_: Border handling.
2. **`nn.MaxPool2d`**: Downsamples data (spatial compression).
3. **`nn.Flatten`**: Converts 3D feature maps to 1D vectors for the linear classifier.


### code: a simple cnn
```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # Image size halves
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)              # Image size halves again
        )
        self.classifier = nn.Linear(32 * 7 * 7, 10) # Assuming 28x28 input

    def forward(self, x):
        x = self.conv_block(x)
        x = torch.flatten(x, 1) # Flatten batch dimensions before passing it into classifier
        return self.classifier(x)
```


## working with text, rnns, attention
### embeddings
Raw text cannot be fed to a network. We convert token IDs to dense vectors using `nn.Embedding`.

```python
# [Vocab_Size, Embedding_Dim]
embedding = nn.Embedding(num_embeddings=10000, embedding_dim=256)
vector = embedding(torch.tensor([42])) # Gets vector for token 42
```

### rnns and grus
Process sequences step-by-step.
```python
# batch_first=True is crucial for modern standard [Batch, Seq, Feature]
rnn = nn.GRU(input_size=256, hidden_size=512, batch_first=True)
output, hidden = rnn(embedded_sequence)
```

### self-attention
The mechanism powering Transformers. The formula is $Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$.
```python
# Q @ K^T / sqrt(d)
scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
weights = torch.softmax(scores, dim=-1)
attention = torch.matmul(weights, value)
```



## customisations

### custom loss functions
You can create custom loss functions with PyTorch— here's an example for a custom PPO objective.

```python
class CustomPPOObjective(nn.Module):
    def __init__(self, clip_ratio=0.2):
        super().__init__()
        self.clip_ratio = clip_ratio

    def forward(self, ratio, advantage):
        # Standard PyTorch operations maintain the graph automatically
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * advantage
        return -torch.min(surr1, surr2).mean() # Negative because we minimize loss
```


### torch distributions

```python
import torch.distributions as dist

# Output of Actor network (logits)
logits = model(state) 

# Create a Categorical distribution (for discrete actions), softmax
m = dist.Categorical(logits=logits)

# Sample an action (Stochastic Policy)
action = m.sample()

# Get log_prob for the loss function (Needed for Policy Gradient)
log_prob = m.log_prob(action)
```


## logging & deployment

### observability (tensorboard / wandb)

It is important to log metrics to visualize loss curves.

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
# Inside training loop
writer.add_scalar("Loss/train", loss.item(), epoch)
writer.add_graph(model, input_to_model)
writer.close()
```

### deployment (torchscript & onnx)

Python is slow. For production, we compile models.

**TorchScript:** Serializes the model to run independent of Python.
```python
model_scripted = torch.jit.script(model)
model_scripted.save("model.pt")
```

**ONNX:** Open standard for interoperability (running PyTorch models in TensorFlow or on Web Browsers).

## additional libraries

- **torchvision / torchtext / torchaudio:** Standard datasets and pre-trained models (ResNet, BERT).
- **pytorch lightning:** Removes boilerplate. It handles the training loop, device placement, and logging automatically.
- **hugging face accelerate:** Scales vanilla PyTorch code to Multi-GPU or TPU with just a few lines of config changes.
- **timm (pytorch image models):** The largest collection of state-of-the-art Computer Vision models.



## cheatsheet

### 1. creation (making stuff)

- `torch.tensor([1, 2, 3])`: Creates a tensor from a list.
- `torch.randn(3, 4)`: Creates random numbers (normal distribution). Used for initializing weights.
- `torch.zeros(3, 4)` / `torch.ones(3, 4)`: Creates masking tensors.
- `torch.arange(0, 10)`: Like Python `range()`. `[0, 1, 2... 9]`

### 2. shapes (the #1 cause of bugs)

- `x.shape`: Not a function, but an attribute. **Must be checked constantly.**    
- `x.view(3, 10)` or `x.reshape(3, 10)`: Changes dimensions.4 (e.g., flatten a 10x10 image into a 100-long vector).
- `x.unsqueeze(0)`: Adds a dimension of size 1. (Turns a vector `[3]` into a batch `[1, 3]`).
- `x.squeeze(0)`: Removes a dimension of size 1.
- `x.permute(0, 2, 1)`: Swaps dimensions. Used to rotate images or matrixes.

### 3. math (doing stuff)

- `torch.matmul(a, b)` or `a @ b`: Matrix multiplication. 
- `x * y`: Element-wise multiplication.
- `x.sum(dim=1)`: Adds up numbers along a specific row/column.
- `x.mean()`: Average.
- `x.argmax(dim=1)`: Returns the **index** of the biggest number.10 Used to turn probability `[0.1, 0.8, 0.1]` into class label `1`.

### 4. neural net blocks

- `nn.Linear(in, out)`: Standard dense layer.
- `nn.Conv2d(...)`: Standard vision layer.
- `nn.ReLU()`: Standard activation function (keeps positives, zeros out negatives).
- `nn.CrossEntropyLoss()`: The standard loss for classification.