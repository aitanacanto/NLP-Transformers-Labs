# Aitana Canto Rico
import torch
import random

import numpy as np

from tqdm import tqdm

detokeniser = ['we', 'they', 'like', 'love', 'computers', 'pizza', 'eh', 'hmm']
subjects = [0, 1]
verbs = [2, 3]
objects = [4, 5]
interjections = [6, 7]

def generate_positions(length:int=8) -> list[int]:
    # Task 1
    # Return a list of subject, verb, and object positions.
    positions = sorted(random.sample(range(length), 3))
    return positions

def generate_input_sequence(positions:list[int], length:int=8) -> list[int]:
    # Task 2
    # Return a list of tokens.
    token_choices = {
        positions[0]: subjects,
        positions[1]: verbs,
        positions[2]: objects
    }

    sequence = [
        random.choice(token_choices[i]) if i in token_choices else random.choice(interjections)
        for i in range(length)
    ]
    return sequence

def compute_target_sequence(input_sequence:list[int]) -> list[int]:
    # Task 3
    return [token for token in input_sequence if token not in interjections]

class ToyTransformer(torch.nn.Module):

    def __init__(self, embed_dim:int=16, num_heads=2, num_context=8):
        super().__init__()
        
        # Task 4.1
        self.semantic = torch.nn.Embedding(8, embed_dim)           # Semantic Embedding -> 8 tokens totally
        self.position = torch.nn.Embedding(num_context, embed_dim) # Positional Embedding -> positions 0-7

        self.attn1 = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4*embed_dim,
            batch_first=True
        )
        self.attn2 = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4*embed_dim,
            batch_first=True
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # Task 4.2
        batch_size, seq_len = x.shape

        semantic_embed = self.semantic(x)                               # (batch, seq_len, embed_dim)
        pos_embed = self.position.weight[:seq_len]                      # (seq_len, embed_dim)
        pos_embed = pos_embed.unsqueeze(0).expand(batch_size, -1, -1)   # (batch, seq_len, embed_dim)

        x_embed = semantic_embed + pos_embed                            # (batch, seq_len, embed_dim)

        x = self.attn1(x_embed) # 1rst Attention Layer
        x = self.attn2(x) # 2nd Attention Layer

        x = x[:, :3, :]  # First 3 tokens -> Subject, Verb, Object

        logits = torch.matmul(x, self.semantic.weight.T)  # (batch, 3, vocab_size)
        return logits
    
def train_toy_transformer(model:ToyTransformer, num_iterations:int=1000, batch_size:int=32):
    model.train()
    optimiser = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    for i in tqdm(range(num_iterations)):
        input_sequence = []
        target_sequence = []
        for _ in range(batch_size):
            positions = generate_positions()
            input_sequence.append(generate_input_sequence(positions))
            target_sequence.append(compute_target_sequence(input_sequence[-1]))
        input_sequence = torch.LongTensor(input_sequence)
        target_sequence = torch.LongTensor(target_sequence)

        # Task 4.3
        output_sequence = model(input_sequence)  # (batch, 3, vocab)
        loss = criterion(output_sequence.view(-1, 8), target_sequence.view(-1))

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if (i + 1) % 100 == 0:
            print('loss:', loss.detach().item(), 'accuracy:', (output_sequence.argmax(-1) == target_sequence).to(float).mean().detach().item())

def validate_toy_transformer(model:ToyTransformer, num_iterations:int=1000, batch_size:int=32) -> float:
    model.eval()
    accuracy = []
    with torch.no_grad():
        for i in range(num_iterations):
                input_sequence = []
                target_sequence = []
                for _ in range(batch_size):
                    positions = generate_positions()
                    input_sequence.append(generate_input_sequence(positions))
                    target_sequence.append(compute_target_sequence(input_sequence[-1]))
                input_sequence = torch.LongTensor(input_sequence)
                target_sequence = torch.LongTensor(target_sequence)
                output_sequence = model(input_sequence)
                accuracy.append((output_sequence.argmax(-1) == target_sequence).to(float).mean().detach().item())
    return sum(accuracy)/len(accuracy)

if __name__ == "__main__":

    # Task 1 test code
    positions = generate_positions()
    print('Task 1:', 'positions:', positions)


    # Task 2 test code
    input_sequence = generate_input_sequence(positions)
    print('Task 2:', 'input sequence:', input_sequence, 'detokenised sequence:', *[detokeniser[i] for i in input_sequence])


    # Task 3 test code
    target_sequence = compute_target_sequence(input_sequence)
    print('Task 3:', 'target sequence:', target_sequence, 'detokenised sequence:', *[detokeniser[i] for i in target_sequence])


    # Task 4 test code
    toy_transformer = ToyTransformer()
    train_toy_transformer(toy_transformer)
    print('Task 4:', 'accuracy:', validate_toy_transformer(toy_transformer))
