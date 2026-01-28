import numpy as np

def dense_net_block(input_data, num_layers, growth_rate, kernels, kernel_size=(3, 3)):
    """
    Performs the forward pass of a DenseNet dense block.
    
    Args:
        input_data: NHWC tensor of shape (N, H, W, C0)
        num_layers: Number of layers in the dense block
        growth_rate: Number of output channels per layer
        kernels: List of kernels, each of shape (kh, kw, C_in, growth_rate)
        kernel_size: Tuple (kh, kw), default (3, 3)
    
    Returns:
        Output tensor of shape (N, H, W, C0 + num_layers * growth_rate)
    """
    N, H, W, C0 = input_data.shape
    
    def relu(x):
        return np.maximum(0.0, x)
    
    def conv2d(X, kernel, stride=1, padding='zero'):
        n_b, n_h, n_w, n_c = X.shape
        k_h, k_w, c_in, c_out = kernel.shape
        if padding == 'zero':
            pad_h = (k_h - 1) // 2
            pad_w = (k_w - 1) // 2
            X_padded = np.pad(X, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
        out = np.zeros((n_b, n_h, n_w, c_out))
        for b in range(n_b):
            for h in range(n_h):
                for w in range(n_w):
                    for c in range(c_out):
                        h_strt = h * stride
                        h_end = h_strt + k_h
                        w_strt = w * stride
                        w_end = w_strt + k_w
                        out[b, h, w, c] = np.sum(X_padded[b, h_strt:h_end, w_strt:w_end, :] * kernel[:, :, :, c])
        return out
    
    out = input_data
    for lay_idx in range(num_layers):
        out = relu(input_data)
        out = conv2d(out, kernels[lay_idx])
        input_data = np.concatenate((input_data, out), axis=-1)
    return input_data

# Test Cases
if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(1, 1, 1, 2)
    kernels = [np.random.randn(3, 3, 2 + i * 1, 1) * 0.01 for i in range(2)]
    print("Test Case 1:")
    print(dense_net_block(X, 2, 1, kernels))
    # Expected: [[[[ 4.96714153e-01, -1.38264301e-01, -2.30186127e-03, -6.70426255e-05]]]]

    np.random.seed(42)
    X = np.random.randn(1, 2, 3, 2)
    kernels = [np.random.randn(3, 3, 2 + i * 1, 1) * 0.01 for i in range(2)]
    print("\nTest Case 2:")
    print(dense_net_block(X, 2, 1, kernels))
