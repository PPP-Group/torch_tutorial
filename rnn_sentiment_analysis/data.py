import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time
import numpy as np
from tqdm import tqdm
from string import punctuation
from collections import Counter
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(123)

review_list = []
label_list = []

for label in ['pos', 'neg']:
    for fname in tqdm(os.listdir(f'./aclImdb/train/{label}/')):
        if 'txt' not in fname:
            continue
        with open(
            os.path.join(f'./aclImdb/train/{label}/', fname), encoding="utf-8"
        ) as f:
            review_list += [f.read()]
            label_list += [label]
print('Number of reviews : ', len(review_list))

review_list = [review.lower() for review in review_list]
review_list = [
    ''.join([letter for letter in review if letter not in punctuation])
    for review in tqdm(review_list)
]

reviews_blob = ' '.join(review_list)

review_words = reviews_blob.split()

count_words = Counter(review_words)

total_review_words = len(review_words)
sorted_review_words = count_words.most_common(total_review_words)

print(sorted_review_words[:10])

vocab_to_token = {
    word: idx+1 for idx, (word, count) in enumerate(sorted_review_words)
}

print(list(vocab_to_token.items())[:10])

reviews_tokenized = []
for review in review_list:
    word_to_token = [vocab_to_token[word] for word in review.split()]
    reviews_tokenized.append(word_to_token)
print(review_list[0])
print(reviews_tokenized[0])

encoded_label_list = [1 if label == 'pos' else 0 for label in label_list]
reviews_len = [len(review) for review in reviews_tokenized]
reviews_tokenized = [
    reviews_tokenized[i] for i, l in enumerate(reviews_len) if l>0
]
encoded_label_list = np.array(
    [encoded_label_list[i] for i, l in enumerate(reviews_len) if l>0],
    dtype='float32'
)

def pad_sequence(reviews_tokenized, sequence_length):
    padded_reviews = np.zeros((len(reviews_tokenized), sequence_length), dtype=int)

    for idx, review in enumerate(reviews_tokenized):
        review_len = len(review)
        if review_len <= sequence_length:
            zeroes = list(np.zeros(sequence_length-review_len))
            new_sequence = zeroes+review
        elif review_len > sequence_length:
            new_sequence = review[0:sequence_length]
        
        padded_reviews[idx,:] = np.array(new_sequence)
    return padded_reviews

sequence_length = 512
padded_reviews = pad_sequence(
    reviews_tokenized=reviews_tokenized, sequence_length=sequence_length
)

plt.hist(reviews_len)

train_val_split = 0.75
train_X = padded_reviews[:int(train_val_split*len(padded_reviews))]
train_y = encoded_label_list[:int(train_val_split*len(padded_reviews))]
validation_X = padded_reviews[int(train_val_split*len(padded_reviews)):]
validation_y = encoded_label_list[int(train_val_split*len(padded_reviews)):]

train_dataset = TensorDataset(
    torch.from_numpy(train_X).to(device), torch.from_numpy(train_y).to(device)
)
validation_dataset = TensorDataset(
    torch.from_numpy(validation_X).to(device), torch.from_numpy(validation_y).to(device)
)
batch_size=32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(
    validation_dataset, batch_size=batch_size, shuffle=True
)

train_data_iter = iter(train_dataloader)
# x_example, y_example = train_data_iter.next()
# print("="*100)
# print('Example Input Size: ', x_example.size())
# print('Example Input:\n', x_example)
# print()
# print('Example Output size: ', y_example.size())
# print('Example Output:\n', y_example)

class RNN(nn.Module):
    def __init__(self, input_dimension, embedding_dimension, hidden_dimension, output_dimension):
        super().__init__()
        self.embedding_layer = nn.Embedding(input_dimension, embedding_dimension)
        self.rnn_layer = nn.RNN(
            embedding_dimension, hidden_dimension, num_layers=1
        )
        self.fc_layer = nn.Linear(hidden_dimension, output_dimension)
    
    def forward(self, sequence):
        embedding=self.embedding_layer(sequence)
        output, hidden_state = self.rnn_layer(embedding)
        final_output = self.fc_layer(hidden_state[-1,:,:].squeeze(0))
        return final_output

input_dimension = len(vocab_to_token)+1
embedding_dimension = 100
hidden_dimension = 32
output_dimension = 1
rnn_model = RNN(
    input_dimension, embedding_dimension, hidden_dimension, output_dimension
)

optim = torch.optim.Adam(rnn_model.parameters())
loss_func = nn.BCEWithLogitsLoss()

rnn_model = rnn_model.to(device)
loss_func = loss_func.to(device)

def accuracy_metric(predictions, ground_truth):
    rounded_predictions = torch.round(torch.sigmoid(predictions))
    success = (rounded_predictions == ground_truth).float()
    accuracy = success.sum() / len(success)
    return accuracy

def train(model, dataloader, optim, loss_func):
    loss = 0
    accuracy = 0
    model.train()
    for sequence, sentiment in dataloader:
        optim.zero_grad()
        preds = model(sequence.T).squeeze()
        loss_curr = loss_func(preds, sentiment)
        accuracy_curr = accuracy_metric(preds, sentiment)
        loss_curr.backward()
        optim.step()
        loss += loss_curr.item()
        accuracy += accuracy_curr.item()
    return loss/len(dataloader), accuracy/len(dataloader)

def validate(model, dataloader, loss_func):
    loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for sequence, sentiment in dataloader:
            preds = model(sequence.T).squeeze()
            loss_curr = loss_func(preds, sentiment)
            accuracy_curr = accuracy_metric(preds, sentiment)
            loss += loss_curr.items()
            accuracy += accuracy_curr.items()
    return loss/len(dataloader), accuracy/len(dataloader)

num_epochs = 10
best_validation_loss = float('inf')

for ep in range(num_epochs):
    time_start = time.time()
    training_loss, train_accuracy = train(
        rnn_model, train_dataloader, optim, loss_func
    )
    validation_loss, validation_accuracy = validate(
        rnn_model, validation_dataloader, loss_func
    )

    time_end = time.time()
    time_delta = time_end - time_start

    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        torch.save(rnn_model.state_dict(), 'rnn_model.pt')
    
    print(f'epoch number: {ep+1} | time elapsed: {time_delta}s')
    print(f'training loss: {training_loss:.3f} | training accuracy: {train_accuracy*100:.2f}%')
    print(f'validation loss: {validation_loss:.3f} | validation accuracy: {validation_accuracy*100:.2f}%')
