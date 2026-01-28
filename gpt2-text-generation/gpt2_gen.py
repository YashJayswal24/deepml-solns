import numpy as np

def layer_norm(x, g, b, eps=1e-9):
    """
    Stabilizes training by standardizing inputs across features.
    
    Args:
        x: Input tensor
        g: Scale parameter (gamma)
        b: Bias parameter (beta)
        eps: Epsilon for numerical stability
    """
    mn = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return g * (x - mn) / np.sqrt(var + eps) + b

def load_encoder_hparams_and_params(model_size: str = "124M", models_dir: str = "models"):
    class DummyBPE:
        def __init__(self):
            self.encoder_dict = {"hello": 1, "world": 2, "<UNK>": 0} # word to token id mapping

        def encode(self, text: str): 
            '''
            returns a list of token ids for the given text
            '''
            tokens = text.strip().split()
            return [self.encoder_dict.get(token, self.encoder_dict["<UNK>"]) for token in tokens] 

        def decode(self, token_ids: list):
            '''
            returns a string for the given list of token ids
            '''
            reversed_dict = {v: k for k, v in self.encoder_dict.items()}
            return " ".join([reversed_dict.get(tok_id, "<UNK>") for tok_id in token_ids])
            
    hparams = {
        "n_ctx": 1024,  # vocabulary context size
        "n_head": 2     # number of attention heads
    }

    params = {
        "wte": np.random.rand(3, 10), # each token id maps to a 10-dim encoding access like wte[token_id]
        "wpe": np.random.rand(1024, 10), # position embeddings access like wpe[position]
        "blocks": [], # list of transformer block parameters
        "ln_f": {
            "g": np.ones(10), # gamma multiplicative factor for layer norm
            "b": np.zeros(10), # beta bias factor for layer norm
        }
    }

    encoder = DummyBPE()
    return encoder, hparams, params

def gen_text(prompt: str, n_tokens_to_generate: int = 40):
    """
    Generates text using a simplified GPT-2 like forward pass.
    """
    encoder, hparams, params = load_encoder_hparams_and_params()
    ans = prompt
    ret = ""
    
    for _ in range(n_tokens_to_generate):
        # Encode current context
        tokens = encoder.encode(ans)[-hparams["n_ctx"]:]
        
        # Add Token Embeddings + Positional Embeddings
        # wte shape: [vocab_size, d_model], wpe shape: [n_ctx, d_model]
        embeddings = params["wte"][tokens] + params["wpe"][np.arange(len(tokens))] # (seq_len, d_model)
        
        # Dummy transformer block processing
        for block in params["blocks"]:
            # Normally, you'd apply attention and feed-forward networks here
            pass
            
        # Layer Normalization
        ln = layer_norm(embeddings, params["ln_f"]["g"], params["ln_f"]["b"]) # (seq_len, d_model)
        
        # Project back to vocabulary (Logits)
        logits = ln @ params["wte"].T # (seq_len, vocab_size)
        
        # Greedy decoding: take the token with highest probability from the last position
        pred_token = np.argmax(logits[-1]) 
        
        # Append to response
        decoded_token = encoder.decode([pred_token])
        ans += " " + decoded_token
        ret += decoded_token + " "
        
    return ret.strip()

# Test Cases
if __name__ == "__main__":
    np.random.seed(42)
    print("Test Case 1:")
    print(gen_text("hello", n_tokens_to_generate=5))
    # Expected: hello hello hello <UNK> <UNK>

    np.random.seed(42)
    print("\nTest Case 2:")
    print(gen_text("hello world", n_tokens_to_generate=10))
    # Expected: world world world world world world world world world world
