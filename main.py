import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- モデル定義 ---
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)  # 入力: 784 -> 出力: 128
        self.fc2 = nn.Linear(128, 10)   # 入力: 128 -> 出力: 10（10クラス分類）

    def forward(self, x):
        # x の形状は (batch, 1, 28, 28) を想定。バッチサイズに合わせてリシェイプ
        x_reshaped = x.view(x.shape[0], -1)  # (batch, 784)
        h = self.fc1(x_reshaped)
        z = torch.sigmoid(h)
        y_hat = self.fc2(z)
        return y_hat

# --- デバイス設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- モデルのロード ---
loaded_model = SimpleMLP().to(device)
# モデルのパラメータをロード（weights_only=True を指定）
loaded_model.load_state_dict(torch.load("modelwithBatch.pth", map_location=device, weights_only=True))
loaded_model.eval()  # 推論モードに設定

st.title("Digit Classification with SimpleMLP")
st.write("画像をアップロードして予測を行います。")

# --- 画像アップロード ---
uploaded_file = st.file_uploader("画像ファイルをアップロードしてください（PNG, JPG, JPEG）", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 画像を読み込み、グレースケールに変換、28x28 にリサイズ
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28, 28))
    st.image(image, caption="アップロード画像", use_column_width=False)

    # PIL Image を NumPy 配列に変換し、正規化（0～1にスケーリング）
    image_np = np.array(image) / 255.0
    # テンソルに変換：形状 [1, 28, 28] → [1, 1, 28, 28]
    image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # --- 予測処理 ---
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = loaded_model(image_tensor)  # 出力形状: [1, 10]
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    st.write("**予測されたクラス:**", predicted_class)
    st.write("**各クラスの確率:**", probabilities.cpu().numpy())

    # --- Matplotlib を用いて画像と予測結果を表示 ---
    fig, ax = plt.subplots()
    ax.imshow(image_tensor.squeeze().cpu().numpy(), cmap="gray")
    ax.set_title(f"Prediction: {predicted_class}")
    ax.axis("off")
    st.pyplot(fig)
else:
    st.write("画像がアップロードされていません。")
