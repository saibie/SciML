import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_circles, make_blobs
from mpl_toolkits.mplot3d import Axes3D 

def generate_data(type='linear'):
    if type == 'linear':
        # 선형적으로 분리 가능한 데이터 (두 개의 Blob)
        X, y = make_blobs(n_samples=100, centers=2, random_state=6, cluster_std=1.0)
        # 클래스 레이블을 -1과 1로 변환 (SVM은 종종 이 형식을 사용)
        y = np.where(y == 0, -1, 1)
        title = "Linearly Separable Data"
    elif type == 'nonlinear':
        # 비선형적으로 분리 가능한 데이터 (동심원)
        X, y = make_circles(n_samples=300, factor=0.5, noise=0.08, random_state=6)
        # 클래스 레이블을 -1과 1로 변환
        y = np.where(y == 0, -1, 1)
        title = "Non-linearly Separable Data (Concentric Circles)"
    else:
        raise ValueError("Data type must be 'linear' or 'nonlinear'")
    return X, y, title

def plot_svm(X, y, model, title):
    plt.figure(figsize=(8, 6))

    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='coolwarm', edgecolors='k', marker='o', label='Data Points')

    ax = plt.gca() # 현재 축 가져오기
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 100)
    yy = np.linspace(ylim[0], ylim[1], 100)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape) # 모델의 결정 함수 값 예측

    plt.contour(XX, YY, Z, colors=['blue', 'black', 'red'], levels=[-1, 0, 1], alpha=0.6,
                linestyles=['--', '-', '--'],
                labels=['Margin (Class 1 side)', 'Decision Boundary', 'Margin (Class -1 side)'])

    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=200, facecolors='none',
                edgecolors='yellow', linewidth=2, label='Support Vectors')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    

def main():
    X_linear, y_linear, title_linear = generate_data(type='linear')

    linear_svm_model = svm.SVC(kernel='rbf', gamma=.001, C=1000.0, random_state=42)
    linear_svm_model.fit(X_linear, y_linear)

    plot_svm(X_linear, y_linear, linear_svm_model, "Linear SVM on " + title_linear)
    

    X_nonlinear, y_nonlinear, title_nonlinear = generate_data(type='nonlinear')
    X_nonlinear += np.array([1, 1])  # 데이터 위치 조정

    nonlinear_svm_model = svm.SVC(kernel='rbf', gamma=1, C=10000.0, random_state=42)
    nonlinear_svm_model.fit(X_nonlinear, y_nonlinear)

    plot_svm(X_nonlinear, y_nonlinear, nonlinear_svm_model, "Kernel SVM (RBF) on " + title_nonlinear)

    X_random = np.random.random((1000, 2)) * 2 - np.array([1, 1])
    y_random = np.where(np.sin(10*X_random[:, 0])/2 - X_random[:, 1] > 0, -1, 1)
    title_random = ""
    
    random_svm_model = svm.SVC(kernel='rbf', gamma=10, C=10000.0, random_state=42)
    random_svm_model.fit(X_random, y_random)

    plot_svm(X_random, y_random, random_svm_model, "Kernel SVM (RBF) on " + title_random)

    X, y = make_circles(n_samples=500, factor=0.3, noise=0.08, random_state=6)
    colors = ['blue' if i == 0 else 'red' for i in y]

    # RBF 하이퍼파라미터
    gamma = 2.0 

    # 원본 X (n_samples, 2)
    x1 = X[:, 0]
    x2 = X[:, 1]

    # 1. 스케일링 팩터 (모든 항에 공통)
    scaling_factor = np.exp(-gamma * (x1**2 + x2**2))


    # --- 2. [Panel 2용] k=0, k=1 블록의 3D 특징 계산 ---
    # (스케일링 팩터를 각 항에 곱해줍니다)
    # Feature 1 (k=0 블록)
    z1_k0 = scaling_factor * 1
    # Feature 2 (k=1 블록, x1 항)
    z2_k1 = scaling_factor * (np.sqrt(2 * gamma) * x1)
    # Feature 3 (k=1 블록, x2 항)
    z3_k1 = scaling_factor * (np.sqrt(2 * gamma) * x2)
    # 3D로 매핑된 데이터 (Panel 2)
    X_3d_k01 = np.vstack((z1_k0, z2_k1, z3_k1)).T

    # --- 3. [Panel 3용] k=2 블록 "만"의 3D 특징 계산 ---
    # (스케일링 팩터를 각 항에 곱해줍니다)
    # Feature 1 (k=2 블록, x1^2 항)
    z1_k2 = scaling_factor * (np.sqrt(2 * gamma**2) * x1**2)
    # Feature 2 (k=2 블록, x1*x2 항)
    z2_k2 = scaling_factor * (2 * gamma * x1 * x2)
    # Feature 3 (k=2 블록, x2^2 항)
    z3_k2 = scaling_factor * (np.sqrt(2 * gamma**2) * x2**2)
    # 3D로 매핑된 데이터 (Panel 3)
    X_3d_k2 = np.vstack((z1_k2, z2_k2, z3_k2)).T


    # --- 4. 시각화 (총 3개 패널) ---
    plt.figure(figsize=(18, 6)) # 가로로 길게

    # --- Panel 1: 원본 2D 데이터 ---
    ax1 = plt.subplot(1, 3, 1)
    ax1.scatter(X[:, 0], X[:, 1], c=colors)
    ax1.set_title('1. Original 2D Data')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')

    # --- Panel 2: k=0, k=1 블록 3D 매핑 ---
    ax2 = plt.subplot(1, 3, 2, projection='3d')
    ax2.scatter(X_3d_k01[:, 0], X_3d_k01[:, 1], X_3d_k01[:, 2], c=colors, s=50, depthshade=True)
    ax2.set_title('2. Mapped 3D (RBF k=0 & k=1)')
    ax2.set_xlabel('z1 (k=0)')
    ax2.set_ylabel('z2 (k=1, x1)')
    ax2.set_zlabel('z3 (k=1, x2)')
    ax2.view_init(elev=30, azim=120) 

    # --- Panel 3: k=2 블록 "만" 3D 매핑 ---
    ax3 = plt.subplot(1, 3, 3, projection='3d')
    ax3.scatter(X_3d_k2[:, 0], X_3d_k2[:, 1], X_3d_k2[:, 2], c=colors, s=50, depthshade=True)
    ax3.set_title('3. Mapped 3D (RBF k=2 only)')
    ax3.set_xlabel('z1 (k=2, x1^2)')
    ax3.set_ylabel('z2 (k=2, x1x2)')
    ax3.set_zlabel('z3 (k=2, x2^2)')
    ax3.view_init(elev=30, azim=105)

    plt.tight_layout()
    
    X = np.random.random((1000, 2)) * 2 - np.array([1, 1])
    y = np.where(np.sin(10*X[:, 0])/2 - X[:, 1] > 0, 0, 1)
    colors = ['blue' if i == 0 else 'red' for i in y]

    # RBF 하이퍼파라미터
    gamma = 2.0 

    # 원본 X (n_samples, 2)
    x1 = X[:, 0]
    x2 = X[:, 1]

    # 1. 스케일링 팩터 (모든 항에 공통)
    scaling_factor = np.exp(-gamma * (x1**2 + x2**2))


    # --- 2. [Panel 2용] k=0, k=1 블록의 3D 특징 계산 ---
    # (스케일링 팩터를 각 항에 곱해줍니다)
    # Feature 1 (k=0 블록)
    z1_k0 = scaling_factor * 1
    # Feature 2 (k=1 블록, x1 항)
    z2_k1 = scaling_factor * (np.sqrt(2 * gamma) * x1)
    # Feature 3 (k=1 블록, x2 항)
    z3_k1 = scaling_factor * (np.sqrt(2 * gamma) * x2)
    # 3D로 매핑된 데이터 (Panel 2)
    X_3d_k01 = np.vstack((z1_k0, z2_k1, z3_k1)).T

    # --- 3. [Panel 3용] k=2 블록 "만"의 3D 특징 계산 ---
    # (스케일링 팩터를 각 항에 곱해줍니다)
    # Feature 1 (k=2 블록, x1^2 항)
    z1_k2 = scaling_factor * (np.sqrt(2 * gamma**2) * x1**2)
    # Feature 2 (k=2 블록, x1*x2 항)
    z2_k2 = scaling_factor * (2 * gamma * x1 * x2)
    # Feature 3 (k=2 블록, x2^2 항)
    z3_k2 = scaling_factor * (np.sqrt(2 * gamma**2) * x2**2)
    # 3D로 매핑된 데이터 (Panel 3)
    X_3d_k2 = np.vstack((z1_k2, z2_k2, z3_k2)).T


    # --- 4. 시각화 (총 3개 패널) ---
    plt.figure(figsize=(18, 6)) # 가로로 길게

    # --- Panel 1: 원본 2D 데이터 ---
    ax1 = plt.subplot(1, 3, 1)
    ax1.scatter(X[:, 0], X[:, 1], c=colors)
    ax1.set_title('1. Original 2D Data')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')

    # --- Panel 2: k=0, k=1 블록 3D 매핑 ---
    ax2 = plt.subplot(1, 3, 2, projection='3d')
    ax2.scatter(X_3d_k01[:, 0], X_3d_k01[:, 1], X_3d_k01[:, 2], c=colors, s=50, depthshade=True)
    ax2.set_title('2. Mapped 3D (RBF k=0 & k=1)')
    ax2.set_xlabel('z1 (k=0)')
    ax2.set_ylabel('z2 (k=1, x1)')
    ax2.set_zlabel('z3 (k=1, x2)')
    ax2.view_init(elev=30, azim=120) 

    # --- Panel 3: k=2 블록 "만" 3D 매핑 ---
    ax3 = plt.subplot(1, 3, 3, projection='3d')
    ax3.scatter(X_3d_k2[:, 0], X_3d_k2[:, 1], X_3d_k2[:, 2], c=colors, s=50, depthshade=True)
    ax3.set_title('3. Mapped 3D (RBF k=2 only)')
    ax3.set_xlabel('z1 (k=2, x1^2)')
    ax3.set_ylabel('z2 (k=2, x1x2)')
    ax3.set_zlabel('z3 (k=2, x2^2)')
    ax3.view_init(elev=30, azim=105)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()