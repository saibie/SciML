import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- 1. 데이터 및 하이퍼파라미터 설정 ---

# A, X: Key
# O, P, Q: Distractor
# E: End (Trigger)
# C, Z: Target
vocab = {'A': 0, 'X': 1, 'O': 2, 'P': 3, 'Q': 4, 'E': 5}
targets = {'C': 0, 'Z': 1}

vocab_size = len(vocab)   # 6
target_size = len(targets) # 2
hidden_size = 10          # 은닉 상태 크기 (작아도 됨)

# *** 핵심: 시퀀스 길이 (얼마나 '장기' 기억이 필요한지) ***
SEQ_LEN = 15 

# --- 2. 데이터 생성기 ---

def generate_data(seq_len):
    """
    [Key, Distractor, ..., Distractor, End] 시퀀스 1개 생성
    """
    key_idx = np.random.choice([0, 1]) # 0 ('A') 또는 1 ('X')
    target_idx = 0 if key_idx == 0 else 1 # 'C' 또는 'Z'
    
    # [Key]
    indices = [key_idx]
    
    # [Distractors]
    distractors = np.random.choice([2, 3, 4], seq_len - 2)
    indices.extend(distractors)
    
    # [End]
    indices.append(5) # 'E'
    
    # 원-핫 인코딩
    # (Batch_size=1, Seq_len, Input_size)
    input_vec = torch.zeros(1, seq_len, vocab_size, dtype=torch.float32)
    for t in range(seq_len):
        input_vec[0, t, indices[t]] = 1
        
    # (Batch_size=1)
    target_tensor = torch.tensor([target_idx], dtype=torch.long)
    
    return input_vec, target_tensor

# --- 3. 모델 정의 ---

class VanillaRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VanillaRNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.W_hy = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # rnn_out: (Batch, Seq_len, Hidden_size)
        # h_n: (Num_layers, Batch, Hidden_size)
        rnn_out, h_n = self.rnn(x)
        
        # 우리는 시퀀스의 '마지막' 스텝의 은닉 상태만 필요
        # rnn_out[:, -1, :] 는 마지막 time step의 hidden state
        y_pred = self.W_hy(rnn_out[:, -1, :])
        return y_pred

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        # nn.RNN을 nn.LSTM으로 교체
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.W_hy = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # lstm_out: (Batch, Seq_len, Hidden_size)
        # h_n: (Num_layers, Batch, Hidden_size) - 단기 기억
        # c_n: (Num_layers, Batch, Hidden_size) - 장기 기억
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 마지막 스텝의 은닉 상태(단기 기억)를 사용
        y_pred = self.W_hy(lstm_out[:, -1, :])
        return y_pred

# --- 4. 학습 함수 ---

def train_model(model, model_name, seq_len):
    print(f"\n--- {model_name} (Seq_len={seq_len}) 학습 시작 ---")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 300
    for epoch in range(epochs):
        model.train()
        
        # 매 에폭마다 새로운 데이터를 생성하여 학습
        input_data, target_data = generate_data(seq_len)
        
        optimizer.zero_grad()
        
        output = model(input_data)
        loss = criterion(output, target_data)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 30 == 0:
            # 간단한 정확도 테스트
            model.eval()
            with torch.no_grad():
                correct = 0
                for _ in range(50): # 50번 테스트
                    test_in, test_tgt = generate_data(seq_len)
                    pred = model(test_in)
                    if torch.argmax(pred, dim=1).item() == test_tgt.item():
                        correct += 1
                accuracy = 100 * correct / 50
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')
                
                # LSTM이 100%에 도달하면 조기 종료
                if accuracy == 100.0:
                    print("100% 도달. 조기 종료.")
                    break

# --- 5. 실행 ---

# 1. 바닐라 RNN 학습
rnn_model = VanillaRNNModel(vocab_size, hidden_size, target_size)
train_model(rnn_model, "Vanilla RNN", SEQ_LEN)

# 2. LSTM 학습
lstm_model = LSTMModel(vocab_size, hidden_size, target_size)
train_model(lstm_model, "LSTM", SEQ_LEN)

