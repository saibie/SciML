import numpy as np
from pydub import AudioSegment
import os
import sys
from matplotlib import pyplot as plt

# --- M4A 파일 로드 헬퍼 (pydub 사용) ---
def load_m4a_to_numpy(file_path):
    """
    M4A 파일을 pydub으로 로드하여 mono numpy 배열로 변환합니다.
    """
    try:
        audio = AudioSegment.from_file(file_path, format="m4a")
        audio = audio.set_channels(1)
        samples = audio.get_array_of_samples()
        np_audio = np.array(samples).astype(np.float32)
        max_val = 2**(audio.sample_width * 8 - 1)
        np_audio /= max_val
        return np_audio, audio.frame_rate
    except FileNotFoundError:
        print(f"오류: {file_path} 파일을 찾을 수 없습니다.", file=sys.stderr)
        return None, None
    except Exception as e:
        print(f"오류: 파일을 로드하는 중 문제가 발생했습니다. FFmpeg가 설치되어 있는지 확인하세요.", file=sys.stderr)
        print(e, file=sys.stderr)
        return None, None

# --- STFT 및 iSTFT 핵심 구현 (Numpy만 사용) ---

def stft(signal, frame_size, hop_size):
    """
    Numpy만을 사용하여 STFT를 계산합니다. (Hann Window 사용)
    """
    window = np.hanning(frame_size)
    num_frames = 1 + int(np.floor((len(signal) - frame_size) / hop_size))
    num_bins = frame_size // 2 + 1
    stft_matrix = np.zeros((num_bins, num_frames), dtype=np.complex128)

    for m in range(num_frames):
        start = m * hop_size
        end = start + frame_size
        frame = signal[start:end] * window
        stft_matrix[:, m] = np.fft.rfft(frame)
        
    return stft_matrix

def istft(stft_matrix, frame_size, hop_size):
    """
    Numpy만을 사용하여 iSTFT (역 STFT)를 구현합니다. (Overlap-Add 방식)
    """
    num_bins, num_frames = stft_matrix.shape
    window = np.hanning(frame_size)
    signal_len = (num_frames - 1) * hop_size + frame_size
    output_signal = np.zeros(signal_len)
    window_sum = np.zeros(signal_len)

    for m in range(num_frames):
        start = m * hop_size
        end = start + frame_size
        frame_spectrum = stft_matrix[:, m]
        frame_time_domain = np.fft.irfft(frame_spectrum, n=frame_size)
        
        output_signal[start:end] += frame_time_domain * window
        window_sum[start:end] += window**2

    window_sum[window_sum < 1e-8] = 1.0
    output_signal /= window_sum
    
    return output_signal

# --- 🌟 Griffin-Lim 알고리즘 구현 🌟 ---

def griffin_lim(spectrogram, frame_size, hop_size, n_iter=50):
    """
    Griffin-Lim 알고리즘을 구현합니다.
    :param spectrogram: 크기(magnitude) 스펙트로그램 (S(m, k))
    :param frame_size: 윈도우 크기 (L)
    :param hop_size: 프레임 이동 간격 (H)
    :param n_iter: 반복 횟수
    :return: 복원된 복소수 STFT 행렬
    """
    print(f"Griffin-Lim 알고리즘 시작 (반복 {n_iter}회)...")
    
    # 1. 초기화: 랜덤 위상 생성
    num_bins, num_frames = spectrogram.shape
    # np.random.rand 범위는 [0, 1), 2*pi 곱하면 [0, 2pi)
    random_phase = np.exp(1j * 2 * np.pi * np.random.rand(num_bins, num_frames))
    
    # S(m,k) * e^(i*phi_0)
    complex_stft = spectrogram * random_phase
    
    # 2. 반복 (Iterate)
    for t in range(n_iter):
        if (t + 1) % 10 == 0:
            print(f"  GLA 반복 {t + 1}/{n_iter}...")
            
        # a. iSTFT: 임시 음성 신호 합성
        temp_signal = istft(complex_stft, frame_size, hop_size)
        
        # b. STFT: 합성된 신호 다시 STFT (일관성 있는 위상 추출)
        # Y_t+1 = STFT(x_t)
        re_stft = stft(temp_signal, frame_size, hop_size)
        
        # c. 투영 (Projection):
        # Y_t+1에서 위상만 가져오고, 크기는 원본 S(m,k)를 사용
        # X_t+1 = S * (Y_t+1 / |Y_t+1|)
        phase = re_stft / (np.abs(re_stft) + 1e-10) # 0으로 나누기 방지
        complex_stft = spectrogram * phase
        
    print("Griffin-Lim 알고리즘 완료.")
    return complex_stft


# --- 스크립트 실행 예제 (Griffin-Lim 적용 및 비교) ---
if __name__ == "__main__":
    
    # --- 설정 ---
    FRAME_SIZE = 1024
    HOP_SIZE = 512     # 50% overlap
    GLA_ITERATIONS = 50 # GLA 반복 횟수
    
    input_file = "1.m4a"

    if not os.path.exists(input_file):
        print(f"'{input_file}'을 찾을 수 없습니다.", file=sys.stderr)
    else:
        # 1. M4A 파일 로드
        print(f"'{input_file}' 로드 중...")
        original_signal, sr = load_m4a_to_numpy(input_file)
        
        if original_signal is not None:
            print(f"로드 완료. 원본 샘플 수: {len(original_signal)}")
            
            # 2. STFT 적용 -> 복소수 스펙트로그램 생성
            print("STFT 계산 중...")
            complex_stft = stft(original_signal, FRAME_SIZE, HOP_SIZE)
            
            # 3. 🔥 스펙트로그램(크기)만 추출 🔥
            # 이것이 우리가 복원에 사용할 유일한 정보입니다.
            spectrogram = np.abs(complex_stft)
            
            # 4. 🔥 Griffin-Lim 알고리즘으로 복소수 스펙트로그램 복원(추정) 🔥
            gla_complex_stft = griffin_lim(spectrogram, FRAME_SIZE, HOP_SIZE, GLA_ITERATIONS)
            
            # 5. 복원된 복소수 스펙트로그램을 iSTFT
            print("Griffin-Lim 결과 iSTFT로 신호 복원 중...")
            reconstructed_signal = istft(gla_complex_stft, FRAME_SIZE, HOP_SIZE)
            
            print(f"복원 완료. 복원된 샘플 수: {len(reconstructed_signal)}")
            
            # 6. 차이 비교
            len_recon = len(reconstructed_signal)
            original_signal_trimmed = original_signal[:len_recon]
            
            difference = original_signal_trimmed - reconstructed_signal
            mse = np.mean(difference**2)
            rmse = np.sqrt(mse)
            
            epsilon = 1e-10
            power_original = np.mean(original_signal_trimmed**2)
            power_noise = mse
            snr = 10 * np.log10(power_original / (power_noise + epsilon))
            
            print("\n--- [Griffin-Lim] 신호 복원 품질 비교 ---")
            print(f" 원본 신호 길이 (비교용): {len(original_signal_trimmed)}")
            print(f"복원 신호 길이: {len(reconstructed_signal)}")
            print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
            print(f"Signal-to-Noise Ratio (SNR): {snr:.2f} dB")
            
            fig, axs = plt.subplots(ncols=2, figsize=(12, 9))
            axs[0].plot(original_signal_trimmed, color='blue', alpha=0.6, label='원본 신호')
            axs[1].plot(reconstructed_signal, color='orange', alpha=0.6, label='복원 신호 (GLA)')
            plt.show()
            
            if snr > 30:
                print("결과: 복원 품질이 매우 높습니다 (비정상적으로 높음).")
            elif snr > 15:
                print("결과: 복원 품질이 좋습니다 (일반적인 GLA 결과).")
            else:
                print("결과: 복원 품질이 낮습니다 (일반적인 GLA 결과, iteration 부족 등).")