
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt

# 日本語フォント指定（Python 3.13 対応）
plt.rcParams['font.family'] = 'IPAexGothic'

st.title(" 商品クラスタリングアプリ（日本語対応）")
st.write("CSVファイルをアップロードして商品のクラスタリングを行います。")

# CSVアップロード
uploaded_file = st.file_uploader("CSVファイルを選択してください", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("###  データプレビュー")
    st.dataframe(df.head())

    # 数値列抽出
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) < 2:
        st.warning("クラスタリングには数値列が2列以上必要です。")
    else:
        # 標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[numeric_cols])

        # クラスタ数選択
        n_clusters = st.slider("クラスタ数を選択", 2, 10, 3)

        # KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        df["Cluster"] = clusters

        st.write("###  クラスタリング結果")
        st.dataframe(df)

        # 2次元散布図
        if X_scaled.shape[1] >= 2:
            fig, ax = plt.subplots()
            scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis')
            ax.set_xlabel(numeric_cols[0])
            ax.set_ylabel(numeric_cols[1])
            ax.set_title("商品クラスタリング結果")
            st.pyplot(fig)

        # Silhouetteスコア
        silhouette_avg = silhouette_score(X_scaled, clusters)
        st.write(f"### シルエットスコア: {silhouette_avg:.3f}")

        # Silhouetteプロット
        fig_sil, ax_sil = plt.subplots()
        sample_silhouette_values = silhouette_samples(X_scaled, clusters)
        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[clusters == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = plt.cm.viridis(float(i) / n_clusters)
            ax_sil.fill_betweenx(
                range(y_lower, y_upper),
                0, ith_cluster_silhouette_values,
                facecolor=color, edgecolor=color, alpha=0.7
            )
            ax_sil.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

        ax_sil.set_xlabel("シルエット係数")
        ax_sil.set_ylabel("クラスタ")
        ax_sil.set_title("Silhouetteプロット")
        ax_sil.axvline(x=silhouette_avg, color="red", linestyle="--")
        st.pyplot(fig_sil)

