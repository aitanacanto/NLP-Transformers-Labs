import torch
import numpy as np


def attention(query:np.ndarray, key:np.ndarray, value:np.ndarray) -> np.ndarray:
    # Task 1
    # Return a 2D output ndarray of shape (L, E) computed from a 2D query ndarray of shape (L, E), a 2D key ndarray of shape (S, E), and a 2D value ndarray of shape (S, E).
    # Formula: A𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛 (𝑄,𝐾,𝑉) = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥 (𝑄𝐾^𝑇) V
    
    # Similarity matrix (Q*K^T) -> similarity of query[i] with key[j]
    similarity_matrix = np.matmul(query, key.transpose())  # (L, S)

    # Softmax row by row -> normalize the similarities
    attention_weights = softmax(similarity_matrix)

    # Multiplication (attention_weights * value)
    output = np.matmul(attention_weights, value)  # (L, E)

    return output


def scaled_attention(query:np.ndarray, key:np.ndarray, value:np.ndarray) -> np.ndarray:
    # Task 2
    # Return a 2D output ndarray of shape (L, E) computed from a 2D query ndarray of shape (L, E), a 2D key ndarray of shape (S, E), and a 2D value ndarray of shape (S, E).
    # Formula: scaled_attn(Q, K, V ) = softmax(QK^T / √E )V
    
    E = query.shape[1]  # Embedding dimension

    # Similarity matrix (Q*K^T) -> similarity of query[i] with key[j]
    similarity_matrix = np.matmul(query, key.T)  # (L, S)

    # Scale Values -> (QK^T) / sqrt(E)
    scaled_similarity = similarity_matrix / np.sqrt(E)

    # Softmax row by row -> normalize the similarities
    attention_weights = softmax(scaled_similarity)

    # Multiplication (attention_weights * value)
    output = np.matmul(attention_weights, value)  # (L, E)

    return output
    

def self_attention(input_tensor:np.ndarray, query_projection:np.ndarray, key_projection:np.ndarray, value_projection:np.ndarray) -> np.ndarray:
    # Task 3
    # Return a 2D output ndarray of shape (L, E) computed from a 2D input ndarray of shape (L, E), a 2D query projection ndarray of shape (E, E), a 2D key projection ndarray of shape (E, E), and a 2D value projection ndarray of shape (E, E).
    # Formula: self_attn(I, W_Q, W_K, W_V) = scaled_attn(IW_Q, IW_K, IW_V)

    # Project the input tensor using the query, key, and value projections
    query = np.matmul(input_tensor, query_projection)  # (L, E)
    key = np.matmul(input_tensor, key_projection)      
    value = np.matmul(input_tensor, value_projection)  

    # Apply scaled attention on the projected query, key, and value
    output = scaled_attention(query, key, value)

    return output


def multi_head_attention(input_tensor:np.ndarray, query_projection:np.ndarray, key_projection:np.ndarray, value_projection:np.ndarray, output_projection:np.ndarray) -> np.ndarray:
    # Task 4
    # Return a 2D output ndarray of shape (L, E) computed from a 2D input ndarray of shape (L, E), a 3D query projection ndarray of shape (H, E, E/H), a 3D key projection ndarray of shape (H, E, E/H), a 3D value projection ndarray of shape (H, E, E/H), and a 2D output projection ndarray of shape (E, E).
    
    H = query_projection.shape[0]  # Number of attention heads
    all_heads = []

    for i in range(H):  # head by head
        head_query = np.matmul(input_tensor, query_projection[i])  # (L, E/H)
        head_key = np.matmul(input_tensor, key_projection[i]) 
        head_value = np.matmul(input_tensor, value_projection[i]) 

        head_output = scaled_attention(head_query, head_key, head_value)  # Apply scaled attention
        all_heads.append(head_output) 

    # Concatenating the outputs of all heads
    concatenated_heads = np.concatenate(all_heads, axis=1)  # Shape (L, H * E/H) = (L, E)

    output = np.matmul(concatenated_heads, output_projection)  # Shape (L, E)

    return output


def compute_gradient(input_tensor:np.ndarray, query_projection:np.ndarray, key_projection:np.ndarray, value_projection:np.ndarray, output_projection:np.ndarray, target_tensor:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = torch.tensor(input_tensor, dtype=torch.float)
    y = torch.tensor(target_tensor, dtype=torch.float)
    mh_attention = torch.nn.MultiheadAttention(query_projection.shape[1], query_projection.shape[0], bias=False)
    mh_attention.in_proj_weight.data = torch.tensor(np.concatenate([query_projection.transpose((0, 2, 1)).reshape(-1, query_projection.shape[1]), key_projection.transpose((0, 2, 1)).reshape(-1, key_projection.shape[1]), value_projection.transpose((0, 2, 1)).reshape(-1, value_projection.shape[1])], 0), dtype=torch.float)
    mh_attention.out_proj.weight.data = torch.tensor(output_projection.transpose(), dtype=torch.float)
    optimiser = torch.optim.SGD(mh_attention.parameters())

    # Task 5
    # Perform forward and backward propagation with a PyTorch MultiheadAttention module.

    y_pred, attention_weights = mh_attention(x, x, x)  # Forward, x = Query, Key & Value
    
    loss = torch.sum((y_pred - y) ** 2)  # Loss = (y_pred - y)^2

    optimiser.zero_grad()  # Clean gradients

    loss.backward()  # Backward

    optimiser.step()  # Backward weigth actualization

    gradient = mh_attention.in_proj_weight.grad.detach().numpy()
    new_projection = mh_attention.in_proj_weight.data.detach().numpy()
    o_gradient = mh_attention.out_proj.weight.grad.detach().numpy()
    new_o_projection = mh_attention.out_proj.weight.data.detach().numpy()
    return y_pred.detach().numpy(), attention_weights.detach().numpy(), gradient, new_projection, o_gradient, new_o_projection


# My auxiliar functions

def softmax(x: np.ndarray) -> np.ndarray: # Conversion to a probability distribution
    """ Softmax row by row """
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Avoid overflow problems
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)   # Normalization

if __name__ == "__main__":

    # Task 1 test code
    print('Task 1:', attention(np.asarray([[0.1, 0.2]]), np.asarray([[2, 3], [3, 2]]), np.asarray([[0, 2], [1, 3]])), sep='\n\n', end='\n\n')

    # Task 2 test code
    print('Task 2:', scaled_attention(np.asarray([[0.1, 0.2]]), np.asarray([[2, 3], [3, 2]]), np.asarray([[0, 2], [1, 3]])), sep='\n\n', end='\n\n')

    # Task 3 test code
    print('Task 3:', self_attention(np.asarray([[1, 2], [2, 1]]), np.asarray([[0.1, 0.2], [0.3, 0.4]]), np.asarray([[2, 3], [1, 2]]), np.asarray([[0, 2], [1, 3]])), sep='\n\n', end='\n\n')

    # Task 4 test code
    print('Task 4:', multi_head_attention(np.asarray([[1, 2], [2, 1]]), np.asarray([[[0.1], [0.3]], [[0.2], [0.4]]]), np.asarray([[[2], [1]], [[3], [2]]]), np.asarray([[[0], [1]], [[2], [3]]]), np.asarray([[3, 4], [5, 6]])), sep='\n\n', end='\n\n')
    
    # Task 5 test code
    y_pred, attention_weights, gradient, new_projection, o_gradient, new_o_projection = compute_gradient(np.asarray([[1, 2], [2, 1]]), np.asarray([[[0.1], [0.3]], [[0.2], [0.4]]]), np.asarray([[[2], [1]], [[3], [2]]]), np.asarray([[[0], [1]], [[2], [3]]]), np.asarray([[3, 4], [5, 6]]), np.zeros((2, 2)))
    print('Task 5:', 'output:', y_pred, 'attention_weights:', attention_weights, 'input projection gradient:', gradient, 'new input projection:', new_projection, 'output projection gradient:', o_gradient, 'new output projection:', new_o_projection, sep='\n\n', end='\n\n')
