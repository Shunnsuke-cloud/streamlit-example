
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import numpy as np

for font in ["Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans JP", "DejaVu Sans"]:
    try:
        plt.rcParams["font.family"] = font
        break
    except:
        pass
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(
    page_title=" 商品クラスタリングアプリ",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title(" 商品クラスタリングアプリ")
st.write("CSVファイルをアップロードして、商品のクラスタリングを行います。")


uploaded_file = st.file_uploader("CSVファイルを選択してください", type=["csv"])

if uploaded_file is not None:
   
    df = pd.read_csv(uploaded_file)
    st.write("###  データプレビュー")
    st.dataframe(df.head())

  
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) < 2:
        st.warning("クラスタリングに必要な数値列が2列以上ありません。")
    else:
      
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[numeric_cols])

     
        n_clusters = st.slider("クラスタ数を選択", 2, 10, 3)

      
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        df["Cluster"] = clusters

        st.write("###  クラスタリング結果")
        st.dataframe(df)

      
        if X_scaled.shape[1] >= 2:
            fig, ax = plt.subplots()
            scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis')
            plt.xlabel(numeric_cols[0])
            plt.ylabel(numeric_cols[1])
            plt.title("Product Clustering Results")
            st.pyplot(fig)

        
        silhouette_avg = silhouette_score(X_scaled, clusters)
        st.write(f"### シルエットスコア: {silhouette_avg:.3f}")

       
        sample_silhouette_values = silhouette_samples(X_scaled, clusters)
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[clusters == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = plt.cm.viridis(float(i) / n_clusters)
            ax2.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7
            )
            ax2.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10  # 次のクラスタの位置

        ax2.set_xlabel("Silhouette Coefficient")
        ax2.set_ylabel("Cluster")
        ax2.set_title("Silhouette Plot")

        ax2.axvline(x=silhouette_avg, color="red", linestyle="--")
        st.pyplot(fig2)
