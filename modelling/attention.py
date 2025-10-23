import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
def attention(q, k, v, mask):
  return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v
