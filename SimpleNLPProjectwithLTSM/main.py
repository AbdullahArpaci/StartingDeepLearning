import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter  #Kelime frekanslarını hesaplamak için
from itertools import product    #Grid search işlemi için kombinasyonlar



text = """Bu ürün beklediğimden çok daha kaliteli çıktı. 
Kargolama hızlıydı ancak paketleme biraz özensizdi.
Fiyatina göre performansı gerçekten tatmin edici.
Renk ve malzeme kalitesi açıklamadakiyle birebir uyumlu.
Tek eksi yanı batarya süresinin biraz kısa olması."""



words= text.replace(".", "").lower().split()
words_count = Counter(words)
vocab = sorted(words_count,key=words_count.get,reverse=True)
word_to_ix = {}
ix_to_word = {}


for i,word in enumerate(vocab):
    word_to_ix[word] = i

for i,word in enumerate(vocab):
    ix_to_word[i] = word


data = [(words[i],words[i+1]) for i in range(len(words)-1)]
print(word_to_ix)
print(ix_to_word)
print(data)



class LSTMmodel(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,hidden_dim)
        self.linear = nn.Linear(hidden_dim,vocab_size)

    def forward(self,x):
        x = self.embedding(x)
        lstm_out,_ = self.lstm(x.view(1,1,-1))
        out = self.linear(lstm_out.view(1,-1))
        return out


model = LSTMmodel(len(vocab),embedding_dim=8,hidden_dim=32)


def prepare_sequence(seq,to_ix):
    return torch.tensor([to_ix[w] for w in seq],dtype=torch.long)


embedding_size = [8,16]
hidden_sizes = [32,64]
learning_rates = [0.01,0.005]
epochs = 50

best_loss = float("inf")
best_params = {}

#grid search

for emb_size,hidden_size,lr in product(embedding_size,hidden_sizes,learning_rates):
    print(f"Değerler: Embedding Size{emb_size}:Hidden Size : {hidden_size}:Learning Rate:{lr}")

    model = LSTMmodel(len(vocab),emb_size,hidden_size)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=lr)

    total_loss = 0
    for epoch in range(epochs):
        epoch_loss = 0
        for word,next_word in data:
            model.zero_grad()


            input_tensor = prepare_sequence([word],word_to_ix)
            target_tensor = prepare_sequence([next_word],word_to_ix)

            output = model(input_tensor)

            loss = loss_function(output,target_tensor)

            loss.backward()
            optimizer.step()
            epoch_loss = loss.item()

        if epoch % 10 == 0:
            print(f"Epoch:{epoch}, Loss: {epoch_loss:.5f}")

        total_loss = epoch_loss
    if total_loss < best_loss:
        best_loss = total_loss
        best_params = {"embedding_data":emb_size,"hidden_size":hidden_size,"learning_rate":lr}


    print(f"Best params: {best_params}")

final_model = LSTMmodel(len(vocab),embedding_dim=best_params["embedding_data"],hidden_dim=best_params["hidden_size"])
optimizer = optim.Adam(final_model.parameters(),lr= best_params["learning_rate"])
loss_function = nn.CrossEntropyLoss()

epochs = 30

for epoch in range(epochs):
    epoch_loss = 0
    for word,next_word in data:
        final_model.zero_grad()

        input_tensor = prepare_sequence([word], word_to_ix)
        target_tensor = prepare_sequence([next_word], word_to_ix)

        output = final_model(input_tensor)

        loss = loss_function(output, target_tensor)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch:{epoch}, Loss: {epoch_loss:.5f}")


#Başlangıç kelimesini ver n adet kelime üretmesini sağla

def predict_sentences(start_word,num_words):
    current_word = start_word
    output_sequence = [current_word]
    for _ in range(num_words):
        with torch.no_grad():
            input_tensor = prepare_sequence([current_word], word_to_ix)
            output = final_model(input_tensor)

            predicted_idx = torch.argmax(output).item()
            current_word = ix_to_word[predicted_idx]
            output_sequence.append(current_word)

    return output_sequence


a = predict_sentences("bu",20)
print(" ".join(a))