import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. 데이터 정의
vocab = ['sunny', 'cloudy', 'rainy']
vocab_size = len(vocab) # 3

# 단어 <-> 인덱스 매핑
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}

# 3. 간단한 학습 데이터 (시퀀스)
# 예: [sunny, cloudy] -> [rainy] (다음날은 rainy)
# 예: [rainy, rainy] -> [cloudy] (다음날은 cloudy)
# (입력 시퀀스, 타겟 날씨)
training_data = [
    ([word_to_idx['sunny'], word_to_idx['cloudy']], word_to_idx['rainy']),
    ([word_to_idx['cloudy'], word_to_idx['rainy']], word_to_idx['sunny']),
    ([word_to_idx['rainy'], word_to_idx['rainy']], word_to_idx['cloudy']),
    ([word_to_idx['cloudy'], word_to_idx['sunny']], word_to_idx['cloudy']),
    ([word_to_idx['sunny'], word_to_idx['sunny']], word_to_idx['cloudy']),
]

# 4. 원-핫 인코딩 유틸리티 함수
def to_one_hot(idx, size):
    """지정된 인덱스를 원-핫 벡터로 변환"""
    vec = torch.zeros(1, size, dtype=torch.float32)
    vec[0, idx] = 1
    return vec

class SimpleVanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleVanillaRNN, self).__init__()

        self.hidden_size = hidden_size
        
        # === 수식의 가중치 행렬 정의 ===
        # W_xh: Input -> Hidden
        self.W_xh = nn.Linear(input_size, hidden_size)
        
        # W_hh: Hidden -> Hidden
        self.W_hh = nn.Linear(hidden_size, hidden_size)
        
        # W_hy: Hidden -> Output
        self.W_hy = nn.Linear(hidden_size, output_size)
        # ================================
        
        # 활성화 함수
        self.tanh = nn.Tanh()

    def forward(self, x_t, h_t_minus_1):
        """
        RNN의 '한 스텝(time step)'을 수식 그대로 계산합니다.
        
        Args:
            x_t (Tensor): 현재 시점 t의 입력 (원-핫 벡터)
            h_t_minus_1 (Tensor): 이전 시점 (t-1)의 은닉 상태
        """
        
        # 1. 은닉 상태 계산: h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
        # (참고: nn.Linear는 bias(b)를 자동으로 포함합니다)
        h_t = self.tanh(self.W_xh(x_t) + self.W_hh(h_t_minus_1))
        
        # 2. 출력 계산: y_t = W_hy * h_t + b_y
        # (CrossEntropyLoss를 쓸 것이므로 softmax는 필요 없습니다)
        y_t_logits = self.W_hy(h_t)
        
        return y_t_logits, h_t

    def init_hidden(self):
        """은닉 상태 h_0 를 0 벡터로 초기화합니다."""
        return torch.zeros(1, self.hidden_size, dtype=torch.float32)

# 하이퍼파라미터 설정
input_size = vocab_size    # 3 (sunny, cloudy, rainy 원-핫)
hidden_size = 5            # 5 (은닉 상태의 차원, 임의로 설정)
output_size = vocab_size   # 3 (다음 날 날씨 예측 결과)
model = SimpleVanillaRNN(input_size, hidden_size, output_size)

def train():
    
    learning_rate = 0.01
    epochs = 100

    # 모델, 손실 함수, 옵티마이저 정의
    criterion = nn.CrossEntropyLoss() # Softmax + NLLLoss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("=== 학습 시작 ===")

    for epoch in range(epochs):
        for seq_indices, target_idx in training_data:
            
            # 1. 그래디언트 초기화
            optimizer.zero_grad()
            
            # 2. h_0 (초기 은닉 상태) 생성
            h_t = model.init_hidden()
            
            # 3. 시퀀스를 한 스텝씩 모델에 입력 (Unfolding)
            #    (W_xh, W_hh, W_hy는 이 루프 내내 공유됩니다)
            for idx in seq_indices:
                # x_t (현재 입력)를 원-핫 벡터로 변환
                x_t = to_one_hot(idx, input_size)
                
                # 모델의 forward 실행
                # y_pred는 현재 스텝의 예측, h_t는 다음 스텝으로 전달될 은닉 상태
                y_pred, h_t = model(x_t, h_t)
            
            # 4. 손실 계산
            #    우리는 시퀀스의 '마지막' 예측(y_pred)만 사용합니다.
            #    target_idx는 LongTensor여야 합니다.
            target = torch.tensor([target_idx], dtype=torch.long)
            loss = criterion(y_pred, target)
            
            # 5. 역전파 및 가중치 업데이트
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    print("=== 학습 완료 ===")

def predict(model, input_seq):
    """학습된 모델로 다음 날 날씨를 예측합니다."""
    # 평가 모드로 설정 (Dropout 등 비활성화)
    model.eval()
    
    # torch.no_grad(): 그래디언트 계산을 중지 (메모리 절약)
    with torch.no_grad():
        # h_0 초기화
        h_t = model.init_hidden()
        
        # 입력 시퀀스를 순차적으로 처리
        print(f"입력: ", end="")
        for idx in input_seq:
            print(f"[{idx_to_word[idx]}] -> ", end="")
            x_t = to_one_hot(idx, input_size)
            # 마지막 은닉 상태(h_t)만 계속 업데이트
            y_pred, h_t = model(x_t, h_t)
        
        # 최종 은닉 상태로 y_pred 예측
        # y_pred (logits)에서 가장 확률이 높은 인덱스(argmax)를 찾음
        predicted_idx = torch.argmax(y_pred, dim=1).item()
        
        print(f"\n예측: [{idx_to_word[predicted_idx]}]")

def test():
    test_seq_1 = [word_to_idx['sunny'], word_to_idx['sunny']]
    predict(model, test_seq_1) 
    # 예상 결과: [cloudy] (학습 데이터 [sunny, sunny] -> [cloudy] 에 따라)

    test_seq_2 = [word_to_idx['cloudy'], word_to_idx['rainy']]
    predict(model, test_seq_2)
    # 예상 결과: [sunny] (학습 데이터 [cloudy, rainy] -> [sunny] 에 따라)

    test_seq_3 = [word_to_idx['cloudy']]
    predict(model, test_seq_3)
    # 예상 결과: [sunny] (학습 데이터 [cloudy, rainy] -> [sunny] 에 따라)

    test_seq_4 = [word_to_idx['cloudy'], word_to_idx['rainy'], word_to_idx['cloudy'], word_to_idx['rainy']]
    predict(model, test_seq_4)
    # 예상 결과: [sunny] (학습 데이터 [cloudy, rainy] -> [sunny] 에 따라)

if __name__ == "__main__":
    train()
    test()