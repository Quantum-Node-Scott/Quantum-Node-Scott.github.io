---
layout: single
title: "#7 Deep Learning & PyTorch Overview"
categories: Bootcamp
tag: [패스트캠퍼스, 패스트캠퍼스AI부트캠프, 업스테이지패스트캠퍼스, UpstageAILab, 국비지원, 패스트캠퍼스업스테이지에이아이랩, 패스트캠퍼스업스테이지부트캠프]
author_profile: false
---


# Introduction to Deep Learning

In this post, we’ll break down the fundamentals of deep learning — the most powerful tool in modern AI. You’ll understand the components that make up deep learning systems, how models learn, and the key mechanisms behind neural network training.

![PyTorch](/assets/images/pytorch.png)

## 1. What is Deep Learning?

Deep learning (DL) is a subfield of machine learning (ML), which in turn is a branch of artificial intelligence (AI).

- AI: Intelligence exhibited by machines

- ML: Algorithms that improve through data

- DL: Neural network-based models that automatically learn hierarchical representations

> DL models can perform complex tasks such as image classification, language translation, and game playing, powered by multiple layers of computation.



## 2. Components of a Deep Learning Pipeline

A successful deep learning system is built on five key components:
- **Data**: Examples include MNIST, CIFAR-10, or text corpora. The format and quality of data depend on the task.
- **Model**: Transforms inputs into desired outputs. Examples include MLP, CNN, RNN.
- **Loss Function**: Measures how well the model is performing (e.g., MSE, Cross-Entropy).
- **Optimization Algorithm**: Adjusts model parameters to minimize loss (e.g., SGD, Adam).
- **Regularization Techniques**: Prevent overfitting (e.g., Dropout, L2 Regularization).

```python
import torch.nn as nn

# Example: Dropout Layer with 0.3 ratio
nn.Dropout(p=0.3)
```
---

## 3. How Models Learn: From Forward to Backward
Deep learning models learn through a two-step process:
1. **Forward Pass**: Input → hidden layers → output. Predictions are made.
2. **Backward Pass (Backpropagation)**: The model calculates gradients using chain rule to update weights.

Key concepts include:
- **Computational Graph**: Represents operations as nodes and edges.
- **Gradient**: Partial derivative of loss with respect to parameters.
- **Chain Rule**: Used to propagate loss backward through layers.

```python
# Pseudocode for simple backward pass
loss.backward()       # Automatically computes gradients
optimizer.step()      # Updates model parameters
```

---

##  4. Backpropagation in Action

To train a neural network, we need to compute the loss gradient for each parameter. This is done efficiently via backpropagation:

Uses the chain rule to break complex derivatives into manageable steps

Allows updating all weights via Stochastic Gradient Descent (SGD) or similar methods

```python
import torch.optim as optim

model = MyModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for batch in dataloader:
    optimizer.zero_grad()
    output = model(batch)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

## 5. Common Architectures: MLP, CNN, RNN

- **MLP (Multi-layer Perceptron)**: Basic feedforward network
```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.layers(x)
```
- **CNN (Convolutional Neural Network)**: Effective for image data
```python
# Example CNN Layer
nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
```
- **RNN (Recurrent Neural Network)**: Used for sequential data like time series or text
```python
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Use last time step
        return out
```


##  6. Enhancing Learning: Techniques and Tricks
- **Dropout**: Randomly deactivates neurons during training

- **Normalization**: Ensures stable and fast training (e.g., BatchNorm)

- **Regularization**: Penalizes model complexity to prevent overfitting

---

## Summary: What You Should Remember

1. Deep learning builds on data, model, loss, optimization, and regularization

2. Forward and backward passes are essential to learning

3. Backpropagation enables gradient-based learning

4. Various architectures and techniques help solve different AI problems