import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import io

# --- 定数 ---
K_RANGE = range(2, 11)

# --- 関数定義 ---

def setup_japanese_font():
    """日本語フォントを設定する"""
    for font in ["Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans JP", "DejaVu Sans"]:
        try:
            plt.rcParams["font.family"] = font
            plt.rcParams['axes.unicode_minus'] = False
            return
        except:
            pass

def load_data(uploaded_file):
    """アップロードされたCSVファイルを読み込む"""
    if uploaded_file:
        try:
            return pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"CSVファイルの読み込み中にエラーが発生しました: {e}")
            return None
    return None

def filter_dataframe(df):
    """データフレームをフィルタリングする"""
    st.sidebar.subheader("2. フィルタ機能")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not categorical_cols:
        st.sidebar.info("フィルタリング可能なカテゴリ列がありません。")
        return df

    filter_col = st.sidebar.selectbox("フィルタリングする列を選択", categorical_cols)
    if filter_col:
        unique_values = df[filter_col].unique().tolist()
        selected_values = st.sidebar.multiselect(f"'{filter_col}' の値を選択", unique_values, default=unique_values)
        return df[df[filter_col].isin(selected_values)]
    return df

def get_clustering_columns(df):
    """クラスタリングに使用する列を選択する"""
    st.sidebar.subheader("3. クラスタリング設定")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("クラスタリングに必要な数値列が2列以上ありません。")
        return None
        
    selected_cols = st.sidebar.multiselect("分析に使用する列を選択", numeric_cols, default=numeric_cols)
    
    if len(selected_cols) < 2:
        st.sidebar.warning("分析には2列以上選択してください。")
        return None
        
    return selected_cols

def estimate_optimal_clusters(X_scaled):
    """エルボー法とシルエットスコアで最適クラスタ数を推定する"""
    st.sidebar.subheader("4. 最適クラスタ数の推定")
    if st.sidebar.button("エルボー法とシルエットスコアで推定"):
        st.session_state.show_estimation = True

    if st.session_state.get('show_estimation', False):
        st.subheader("最適クラスタ数の推定結果")
        with st.spinner("計算中..."):
            distortions, silhouette_scores = [], []
            for k in K_RANGE:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                kmeans.fit(X_scaled)
                distortions.append(kmeans.inertia_)
                silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

            col1, col2 = st.columns(2)
            with col1:
                fig_elbow, ax_elbow = plt.subplots()
                ax_elbow.plot(K_RANGE, distortions, marker='o')
                ax_elbow.set_xlabel("クラスタ数 (k)")
                ax_elbow.set_ylabel("SSE (歪み)")
                ax_elbow.set_title("エルボー法による最適クラスタ数の検討")
                st.pyplot(fig_elbow)
                st.write("グラフの「肘」のように見える部分が最適なクラスタ数の候補です。")
            
            with col2:
                fig_sil, ax_sil = plt.subplots()
                ax_sil.plot(K_RANGE, silhouette_scores, marker='o')
                ax_sil.set_xlabel("クラスタ数 (k)")
                ax_sil.set_ylabel("シルエットスコア")
                ax_sil.set_title("シルエットスコアによる最適クラスタ数の検討")
                st.pyplot(fig_sil)
                best_k_by_silhouette = K_RANGE[np.argmax(silhouette_scores)]
                st.write(f"シルエットスコアが最も高いクラスタ数は **{best_k_by_silhouette}** です。")

def run_clustering(X_scaled, n_clusters):
    """K-meansクラスタリングを実行する"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(X_scaled)
    return clusters, kmeans

def display_results(df_original, df_clustered, X_scaled, selected_cols, clusters, n_clusters):
    """結果を表示・ダウンロードする"""
    st.subheader("データプレビュー")
    st.dataframe(df_original.head())

    tab1, tab2, tab3 = st.tabs(["クラスタリング結果", "結果の可視化", "シルエット分析"])

    with tab1:
        st.subheader("クラスタリング結果")
        st.dataframe(df_clustered)
        csv = df_clustered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="結果をCSVでダウンロード",
            data=csv,
            file_name='clustered_data.csv',
            mime='text/csv',
        )

    with tab2:
        if X_scaled.shape[1] >= 2:
            st.subheader("クラスタリング結果の可視化")
            fig_scatter, ax_scatter = plt.subplots()
            ax_scatter.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis')
            ax_scatter.set_xlabel(selected_cols[0])
            ax_scatter.set_ylabel(selected_cols[1])
            ax_scatter.set_title("Product Clustering Results")
            st.pyplot(fig_scatter)

            img_buf = io.BytesIO()
            fig_scatter.savefig(img_buf, format='png')
            img_buf.seek(0)
            st.download_button(
                label="散布図をダウンロード",
                data=img_buf,
                file_name="clustering_scatter.png",
                mime="image/png"
            )

    with tab3:
        st.subheader("シルエット分析")
        silhouette_avg = silhouette_score(X_scaled, clusters)
        st.write(f"平均シルエットスコア: **{silhouette_avg:.3f}**")

        sample_silhouette_values = silhouette_samples(X_scaled, clusters)
        fig_sil_plot, ax_sil_plot = plt.subplots(figsize=(8, 4))
        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[clusters == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = plt.cm.viridis(float(i) / n_clusters)
            ax_sil_plot.fill_betweenx(
                np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                facecolor=color, edgecolor=color, alpha=0.7
            )
            ax_sil_plot.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

        ax_sil_plot.set_xlabel("Silhouette Coefficient")
        ax_sil_plot.set_ylabel("Cluster")
        ax_sil_plot.set_title("Silhouette Plot")
        ax_sil_plot.axvline(x=silhouette_avg, color="red", linestyle="--")
        st.pyplot(fig_sil_plot)

# --- Streamlit App ---

def main():
    """メインのアプリケーション"""
    st.set_page_config(page_title="商品クラスタリングアプリ", layout="wide")
    setup_japanese_font()

    st.title("商品クラスタリングアプリ")
    st.write("CSVファイルをアップロードして、商品のクラスタリングを行います。")

    st.sidebar.header("操作パネル")
    uploaded_file = st.sidebar.file_uploader("1. CSVファイルを選択", type=["csv"])

    if uploaded_file:
        if 'df_original' not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
            st.session_state.df_original = load_data(uploaded_file)
            st.session_state.uploaded_file_name = uploaded_file.name
            # 新しいファイルがアップロードされたら、推定結果の表示フラグをリセット
            if 'show_estimation' in st.session_state:
                del st.session_state.show_estimation

        if st.session_state.df_original is not None:
            df_filtered = filter_dataframe(st.session_state.df_original.copy())
            
            selected_cols = get_clustering_columns(df_filtered)

            if selected_cols:
                X = df_filtered[selected_cols]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                estimate_optimal_clusters(X_scaled)

                n_clusters = st.sidebar.slider("クラスタ数を選択", min_value=2, max_value=10, value=3)

                clusters, _ = run_clustering(X_scaled, n_clusters)
                
                df_clustered = df_filtered.copy()
                df_clustered["Cluster"] = clusters

                display_results(st.session_state.df_original, df_clustered, X_scaled, selected_cols, clusters, n_clusters)
    else:
        st.info("サイドバーからCSVファイルをアップロードしてください。")
        # ファイルがクリアされたらセッションステートをリセット
        for key in list(st.session_state.keys()):
            del st.session_state[key]


if __name__ == "__main__":
    main()