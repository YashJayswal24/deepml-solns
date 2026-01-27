import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size):
        """
        Initializes the LSTM with random weights and zero biases.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weight matrices for input gate, forget gate, candidate cell state, and output gate
        # Each weight matrix is (hidden_size, input_size + hidden_size)
        # We concatenate input and hidden state for simpler computation
        self.W_i = np.random.randn(hidden_size, input_size + hidden_size)
        self.W_f = np.random.randn(hidden_size, input_size + hidden_size)
        self.W_g = np.random.randn(hidden_size, input_size + hidden_size)
        self.W_o = np.random.randn(hidden_size, input_size + hidden_size)

        # Biases initialized to zero
        self.b_i = np.zeros((hidden_size, 1))
        self.b_f = np.zeros((hidden_size, 1))
        self.b_g = np.zeros((hidden_size, 1))
        self.b_o = np.zeros((hidden_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x, initial_hidden_state, initial_cell_state):
        """
        Processes a sequence of inputs and returns the hidden states at each time step,
        as well as the final hidden state and cell state.
        
        x: Input sequence of shape (seq_len, input_size)
        initial_hidden_state: (hidden_size, 1) or similar
        initial_cell_state: (hidden_size, 1) or similar
        """
        h = initial_hidden_state
        c = initial_cell_state
        hidden_states = []

        for x_t in x:
            # Reshape x_t to (input_size, 1)
            x_t = x_t.reshape(-1, 1)
            
            # Concatenate input and previous hidden state
            combined = np.vstack((h, x_t))

            # Compute gates
            i_t = self.sigmoid(np.dot(self.W_i, combined) + self.b_i)
            f_t = self.sigmoid(np.dot(self.W_f, combined) + self.b_f)
            g_t = np.tanh(np.dot(self.W_g, combined) + self.b_g)
            o_t = self.sigmoid(np.dot(self.W_o, combined) + self.b_o)

            # Update cell state
            c = f_t * c + i_t * g_t
            
            # Update hidden state
            h = o_t * np.tanh(c)
            
            hidden_states.append(h)

        hidden_states = np.array(hidden_states)
        return hidden_states, h, c

# Test Case 1
import numpy as np

input_sequence = np.array([[1.0], [2.0], [3.0]])
initial_hidden_state = np.zeros((1, 1))
initial_cell_state = np.zeros((1, 1))

lstm = LSTM(input_size=1, hidden_size=1)
# Set weights and biases for reproducibility
lstm.Wf = np.array([[0.5, 0.5]])
lstm.Wi = np.array([[0.5, 0.5]])
lstm.Wc = np.array([[0.3, 0.3]])
lstm.Wo = np.array([[0.5, 0.5]])
lstm.bf = np.array([[0.1]])
lstm.bi = np.array([[0.1]])
lstm.bc = np.array([[0.1]])
lstm.bo = np.array([[0.1]])

outputs, final_h, final_c = lstm.forward(input_sequence, initial_hidden_state, initial_cell_state)

print(final_h)

# Expected Output:
# [[0.73698596]]
