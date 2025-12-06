import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error



"""
generate_data:
- Sinüs dalgası üretir
- Bu dalgayı zaman serisi şeklinde parçalara böler
- 50 değer -> bir sonraki değeri tahmin etme şeklinde dataset oluşturur
"""
def generate_data(seq_length=50, num_samples=1000):

    # 0 ile 100 arasında 1000 adet eşit aralıklı sayı üretir
    X = np.linspace(0, 100, num_samples)

    # Üretilen X değerlerinin sinüsü alınır → sinüs dalgası oluşturur
    y = np.sin(X)

    sequence = []   # Modelin girişi olan 50 uzunluklu diziler
    target = []     # Her 50'lik dizinin bir sonraki (51.) değeri

    for i in range(len(X) - seq_length):
        # 50 değerlik pencereyi ekler
        sequence.append(y[i:i + seq_length])

        # Bu pencerenin hemen sonrasındaki değeri hedef olarak ekler
        target.append(y[i + seq_length])

    # Sinüs dalgasını grafik olarak çizdirir
    plt.figure(figsize=(10, 8))
    plt.plot(X, y, label="sin(t)", color="b", linewidth=2)
    plt.title("Sinüs Dalgası")
    plt.xlabel("Zaman (radyan)")
    plt.ylabel("Genlik")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Python listelerini numpy dizisine çevirip geri döndürür
    return np.array(sequence), np.array(target)





"""
PyTorch RNN modeli:
- input_size: Her zaman adımında giren veri boyutu (sinüs → 1 değer)
- hidden_state: RNN içindeki nöron sayısı
- output_size: Tek bir değer tahmin edeceğimiz için = 1
"""
class RNN(nn.Module):
    def __init__(self, input_size, hidden_state, output_size, num_layers=1):
        super().__init__()

        # batch_first=True → (batch, seq_len, input_size) formatını kabul etmesini sağlar
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_state,
            num_layers=num_layers,
            batch_first=True
        )

        # RNN'in çıktısını (hidden_state boyutunu) tek bir değere dönüştürür
        self.fc = nn.Linear(hidden_state, output_size)

    def forward(self, x):
        # RNN çıktılarını alır
        # rnn_out: tüm zaman adımlarının çıktısı
        # hidden: son gizli durum (kullanmayı tercih etmiyoruz)
        rnn_out, hidden = self.rnn(x)

        # Son zaman adımının çıktısını alır → (batch, hidden_size)
        last_step_out = rnn_out[:, -1, :]

        # Bu çıktıyı tam bağlı katmandan geçirerek tahmin üretir
        prediction = self.fc(last_step_out)

        return prediction



# -------------------------
# Model Hiperparametreleri
# -------------------------
seq_length = 50        # 50 geçmiş değer → 51. değer tahmini
input_size = 1         # Sinüs dalgası → tek boyutlu veri
hidden_size = 32     # RNN içindeki nöron sayısı
output_size = 1        # Tek bir tahmin çıkışı
num_layers = 1        # Tek katmanlı RNN
epochs = 20          # Eğitim tekrar sayısı
batch_size = 32        # Her eğitim adımında kullanılacak örnek sayısı
learning_rate = 0.001  # Adam optimizer için ideal LR



# Veri oluşturulur
X, y = generate_data(seq_length)

# PyTorch tensörlerine dönüştürülür
# unsqueeze(-1) → (batch, seq_len, 1) formatı sağlar
X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

# PyTorch Dataset ve DataLoader oluşturulur
dataset = torch.utils.data.TensorDataset(X, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)



# Model, Loss fonksiyonu ve optimizer tanımlanır
Model = RNN(input_size, hidden_size, output_size, num_layers)
Loss = nn.MSELoss()
optimizer = optim.Adam(Model.parameters(), lr=learning_rate)



"""
train_model:
- Modeli eğitir
- Her epoch sonunda Loss ve R2 skorunu hesaplar
- Eğitim bittikten sonra Loss ve R2 grafiklerini çizer
"""
def train_model(epochs, model, dataloader, loss_function, optimizer):

    model.train()  # Modeli eğitim moduna al

    epoch_losses = []      # Her epoch için Loss kaydı
    epoch_r2_scores = []   # Her epoch için R2 skoru kaydı

    for epoch in range(epochs):
        total_loss = 0
        all_predictions = []
        all_targets = []

        for batch_x, batch_y in dataloader:

            optimizer.zero_grad()  # Gradyanları sıfırla

            # 1. Forward Pass → Model tahmin yapar
            outputs = model(batch_x)

            # 2. Loss hesaplanır
            loss = loss_function(outputs, batch_y)

            # 3. Backward Pass → Gradyanlar hesaplanır
            loss.backward()

            # 4. Parametreler güncellenir
            optimizer.step()

            total_loss += loss.item()

            # R2 skoru için tahmin ve gerçekleri sakla
            all_predictions.append(outputs.detach().numpy())
            all_targets.append(batch_y.detach().numpy())

        # Epoch ortalama Loss
        avg_loss = total_loss / len(dataloader)

        # Tüm batch'leri birleştir
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)

        # R2 skoru hesapla
        current_r2 = r2_score(all_targets, all_predictions)

        # Kayıt et
        epoch_losses.append(avg_loss)
        epoch_r2_scores.append(current_r2)

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f} - R2 Score: {current_r2:.4f}")



    # -----------------------
    # Eğitim Sonu Grafikler
    # -----------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss Grafiği
    ax1.plot(range(1, epochs + 1), epoch_losses, marker="o")
    ax1.set_title("Loss (MSE) - Hata Oranı")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.grid(True)

    # R2 Score Grafiği
    ax2.plot(range(1, epochs + 1), epoch_r2_scores, marker="o")
    ax2.set_title("R2 Score (Başarım)")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("R2 Score")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def test_model(model):
    # 1. MODELİ TEST MODUNA AL
    model.eval()
    # (Dropout veya Batch Norm varsa kapatır, şu anki modelinde yok ama el alışkanlığı olsun)

    # 2. TEST VERİSİ OLUŞTUR
    # Eğitimde kullandığımızdan farklı, kısa bir aralık oluşturalım
    X_test_raw, y_test_raw = generate_data(seq_length=50, num_samples=1000)

    # Tensor'a çevir
    X_test_tensor = torch.tensor(X_test_raw, dtype=torch.float32).unsqueeze(-1)

    # 3. TAHMİN YAP (GRADYAN HESAPLAMADAN)
    with torch.no_grad():
        # Modelden tahmin iste
        predictions = model(X_test_tensor)

        # Sonuçları grafik çizdirmek için Numpy formatına geri çevir
        predictions = predictions.numpy().flatten()

    # 4. GÖRSELLEŞTİRME
    plt.figure(figsize=(12, 6))

    # Gerçek Veri (Mavi)
    limit = 200
    plt.plot(y_test_raw[:limit], label='Gerçek (Ground Truth)', color='blue', linewidth=2)
    plt.plot(predictions[:limit], label='Model Tahmini (Predictions)', color='red', linestyle='--', linewidth=2)

    plt.title("Gerçek vs Tahmin (Düzeltilmiş Örnekleme Hızı)")
    plt.xlabel("Zaman Adımları")
    plt.ylabel("Genlik")
    plt.legend()
    plt.grid(True)
    plt.show()

# Modeli eğitir
train_model(epochs, Model, dataloader, Loss, optimizer)


print("Test başlatılıyor...")
test_model(Model)