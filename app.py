# これを使ってください
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.title("🛒 商品クラスタリングアプリ")

st.write("CSVファイルをアップロードして、商品のクラスタリングを行います。")

# ファイルアップロード
uploaded_file = st.file_uploader("CSVファイルを選択してください", type=["csv"])

if uploaded_file is not None:
    # CSV読み込み
    df = pd.read_csv(uploaded_file)
    st.write("### 📊 データプレビュー")
    st.dataframe(df.head())

    # 数値列を抽出
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) < 2:
        st.warning("クラスタリングに必要な数値列が2列以上ありません。")
    else:
        # 標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[numeric_cols])

        # クラスタ数の選択
        n_clusters = st.slider("クラスタ数を選択", 2, 10, 3)

        # KMeans実行
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        # 結果追加
        df["Cluster"] = clusters
        st.write("### 🧩 クラスタリング結果")
        st.dataframe(df)

        # グラフ表示（2次元）
        if X_scaled.shape[1] >= 2:
            fig, ax = plt.subplots()
            scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis')
            plt.xlabel(numeric_cols[0])
            plt.ylabel(numeric_cols[1])
            plt.title("商品クラスタリング結果")
            st.pyplot(fig)
