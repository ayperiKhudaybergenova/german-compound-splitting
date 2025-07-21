# %%
def read_train_data(filepath):
    
    words = []
    labels = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()  
            if not parts:
                continue
            word = "".join(parts) 
            label = []
            for part in parts:
                label.append("B")
                label.extend(["I"] * (len(part) - 1))
            assert len(word) == len(label)
            words.append(list(word))     
            labels.append(label)        

    return words, labels


# %%
train_path = "nlp2-a3.train"
char_sequences, label_sequences = read_train_data(train_path)


# %%
from collections import Counter

char_counter = Counter(char for seq in char_sequences for char in seq)
chars = sorted(char_counter)

char2idx = {char: idx + 1 for idx, char in enumerate(chars)}
char2idx["<PAD>"] = 0
char2idx["<UNK>"] = len(char2idx)  


label2idx = {"B": 0, "I": 1, "<PAD>": 2}
idx2label = {0: "B", 1: "I", 2: "<PAD>"}


# %%
def encode_and_pad(sequences, mapping, pad_value, max_len):
    encoded = []
    for seq in sequences:
        ids = [mapping[char] for char in seq]
        padded = ids + [pad_value] * (max_len - len(ids))
        encoded.append(padded)
    return encoded


# %%

max_len = max(len(seq) for seq in char_sequences)


X = encode_and_pad(char_sequences, char2idx, pad_value=char2idx["<PAD>"], max_len=max_len)
y = encode_and_pad(label_sequences, label2idx, pad_value=label2idx["<PAD>"], max_len=max_len)


# %%
import torch
from torch.utils.data import TensorDataset, DataLoader


X_tensor = torch.tensor(X, dtype=torch.long)
y_tensor = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)


# %%
import torch.nn as nn

class CharBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, padding_idx):
        super(CharBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)                   
        lstm_out, _ = self.lstm(embedded)             
        logits = self.fc(lstm_out)                     
        return logits


# %%
model = CharBiLSTM(
    vocab_size=len(char2idx),
    embedding_dim=64,
    hidden_dim=128,
    output_dim=len(label2idx),
    padding_idx=char2idx["<PAD>"]
)


# %%
import torch.optim as optim

criterion = nn.CrossEntropyLoss(ignore_index=label2idx["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = criterion.to(device)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        output = model(batch_x) 

        output = output.view(-1, output.shape[-1])
        batch_y = batch_y.view(-1)

        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")


# %%
def read_test_data(filepath):
    words = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word:
                words.append(list(word)) 
    return words

# %%
test_sequences = read_test_data("nlp2-a3.test")


# %%
model.eval()
predictions = []

with torch.no_grad():
    for seq in test_sequences:
        seq_ids = [char2idx.get(c, char2idx["<UNK>"]) for c in seq]
        seq_tensor = torch.tensor(seq_ids, dtype=torch.long).unsqueeze(0).to(device) 
        output = model(seq_tensor)                 
        pred = torch.argmax(output, dim=-1).squeeze(0).tolist()
        predictions.append(pred)


# %%
def reconstruct_word(chars, labels):
    result = []
    for c, label in zip(chars, labels):
        if label2idx["B"] == label and result:
            result.append(" ")
        result.append(c)
    return "".join(result)

segmented_words = [reconstruct_word(word, pred) for word, pred in zip(test_sequences, predictions)]



# %%
with open("a3-rnn.predictions", "w", encoding="utf-8") as f:
    for word in segmented_words:
        f.write(word + "\n")



