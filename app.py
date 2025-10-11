
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import japanize_matplotlib
import matplotlib.cm as cm

st.set_page_config(page_title=" 商品クラスタリング（Silhouette可視化）", layout="wide")


st.markdown("<h1 style='text-align:center; margin-bottom:6px;'>🛒 商品クラスタリングアプリ</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#4b5563; margin-top:0;'>CSVをアップロードして数値列を選択、k-meansでクラスタリング。SilhouetteスコアとSilhouetteプロットで評価します。</p>", unsafe_allow_html=True)
st.write("---")


with st.container():
    left_col, right_col = st.columns([2, 1], gap="large")

    with left_col:
        uploaded_file = st.file_uploader("📤 CSVファイルを選択してください", type=["csv"])
        st.write("ヒント: 最低でも数値列が2列必要です（クラスタリング＋可視化のため）。")

    with right_col:
        n_clusters = st.slider("🔢 クラスタ数（k）", 2, 10, 3)
        run_btn = st.button("▶ クラスタリング実行")

st.write("---")


def prepare_data(df, selected_cols):
    X = df[selected_cols].astype(float).copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def compute_kmeans(X_scaled, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X_scaled)
    return kmeans, labels

def silhouette_summary(X_scaled, labels):
    
    score = silhouette_score(X_scaled, labels)
    return score

def plot_silhouette(X_scaled, labels, n_clusters):
    
    silhouette_vals = silhouette_samples(X_scaled, labels)
    y_lower = 10
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = cm.get_cmap("nipy_spectral")
    for i in range(n_clusters):
        ith_cluster_vals = silhouette_vals[labels == i]
        ith_cluster_vals.sort()
        size_cluster_i = ith_cluster_vals.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cmap(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_vals,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, f"Cluster {i}", fontsize=9, va='center')

        y_lower = y_upper + 10  

    ax.set_xlabel("Silhouette coefficient 値")
    ax.set_ylabel("クラスタごとのサンプル")
    ax.set_yticks([])  
    ax.set_xlim([-0.1, 1])
    ax.set_title("Silhouette Plot（クラスタ別）")
    ax.axvline(x=np.mean(silhouette_vals), color="red", linestyle="--", label=f"平均 = {np.mean(silhouette_vals):.3f}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig

def plot_scatter_pca(X_scaled, labels, selected_cols):
    
    if X_scaled.shape[1] >= 2:
        pca = PCA(n_components=2)
        X_2 = pca.fit_transform(X_scaled)
        pc1_name = "PC1"
        pc2_name = "PC2"
    else:
        
        X_2 = np.hstack([X_scaled, np.zeros((X_scaled.shape[0], 1))])
        pc1_name = selected_cols[0]
        pc2_name = ""

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = cm.get_cmap("tab10")
    for i in np.unique(labels):
        idx = labels == i
        ax.scatter(X_2[idx, 0], X_2[idx, 1], s=80, alpha=0.7, label=f"Cluster {i}", color=cmap(int(i) % 10))
    ax.set_xlabel(pc1_name)
    ax.set_ylabel(pc2_name if pc2_name else "")
    ax.set_title("クラスタ別散布図（PCA2次元）")
    ax.legend()
    fig.tight_layout()
    return fig


if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"CSVの読み込みでエラーが発生しました: {e}")
        st.stop()

    st.markdown("### データプレビュー")
    st.dataframe(df.head())

    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        st.warning("このCSVには数値列が含まれていません。数値列が必要です。")
        st.stop()

    st.markdown("###  クラスタリング対象の数値列を選択してください")
    selected_cols = st.multiselect("数値列を複数選択（少なくとも2列推奨）", options=numeric_cols, default=numeric_cols[:2])

    if len(selected_cols) < 1:
        st.warning("少なくとも1つの数値列を選択してください。")
        st.stop()

    
    if run_btn:
        
        if len(selected_cols) < 1:
            st.error("数値列を1つ以上選択してください。")
            st.stop()

        
        try:
            X_scaled, scaler = prepare_data(df, selected_cols)
        except Exception as e:
            st.error(f"データの前処理に失敗しました: {e}")
            st.stop()

        
        if n_clusters >= X_scaled.shape[0]:
            st.error("クラスタ数 k がサンプル数以上です。k を小さくしてください。")
            st.stop()

        
        try:
            kmeans, labels = compute_kmeans(X_scaled, n_clusters)
        except Exception as e:
            st.error(f"KMeans 実行時にエラーが発生しました: {e}")
            st.stop()

        
        try:
            sil_score = silhouette_score(X_scaled, labels)
        except Exception as e:
            st.error(f"Silhouette スコアの計算でエラー: {e}")
            sil_score = None

        
        st.markdown("### 結果サマリ")
        cols = st.columns([1, 2])
        with cols[0]:
            if sil_score is not None:
                
                if sil_score >= 0.5:
                    comment = "✅ 良好なクラスタリングです！"
                elif sil_score >= 0.25:
                    comment = "そこそこ分かれていますが改善の余地があります。"
                else:
                    comment = "クラスタがうまく分かれていない可能性があります。"

                st.metric(label="Silhouette スコア（平均）", value=f"{sil_score:.3f}", delta=None)
                st.caption(comment)
            else:
                st.write("Silhouette スコアを計算できませんでした。")

        with cols[1]:
            st.write("####  選択した特徴量")
            st.write(", ".join(selected_cols))

        
        df_result = df.copy()
        df_result["Cluster"] = labels
        st.markdown("### クラスタリング結果（先頭 20 行）")
        st.dataframe(df_result.head(20))

    
        st.markdown("### 可視化")
        fig1 = None
        fig2 = None
        try:
            fig1 = plot_silhouette(X_scaled, labels, n_clusters)
        except Exception as e:
            st.warning(f"Silhouetteプロットの描画に失敗しました: {e}")
        try:
            fig2 = plot_scatter_pca(X_scaled, labels, selected_cols)
        except Exception as e:
            st.warning(f"散布図の描画に失敗しました: {e}")

        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            if fig1 is not None:
                st.pyplot(fig1)
        with viz_col2:
            if fig2 is not None:
                st.pyplot(fig2)

        st.success("クラスタリングが完了しました。必要ならクラスタ数を変更して再実行してください。")

else:
    st.info("まずはCSVファイルをアップロードして、数値列を選択してください。")
