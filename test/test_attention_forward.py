import torch

from modelling.attention import Attention

def test_attention_forward():
    torch.manual_seed(0)
    q = torch.randn(1, 3, 4)
    k = torch.randn(1, 3, 4)
    v = torch.randn(1, 3, 4)

    # manual computation
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / d_k**0.5
    weights = torch.softmax(scores, dim=-1)
    expected = torch.matmul(weights, v)

    # module
    attn = Attention(mask_future=False)
    actual = attn(q, k, v)

    assert torch.allclose(expected, actual, atol=1e-6)
