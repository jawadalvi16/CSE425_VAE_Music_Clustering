import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from collections import Counter


# -----------------------------
# Paths
# -----------------------------
AUDIO_PATH = "data/audio_mfcc.npy"
LYRICS_PATH = "data/lyrics_emb.npy"
Z_PATH = "data/latent_z.npy"
GENRES_PATH = "data/genres.npy"

K = 10  # GTZAN has 10 genres


# -----------------------------
# Helpers
# -----------------------------
def purity_score(y_true, y_pred):
    total = 0
    for c in np.unique(y_pred):
        idx = np.where(y_pred == c)[0]
        labels = y_true[idx]
        most_common = Counter(labels).most_common(1)[0][1]
        total += most_common
    return total / len(y_true)


def clustering_metrics(X, y_true, y_pred):
    # Internal metrics (no labels needed)
    sil = silhouette_score(X, y_pred)
    ch = calinski_harabasz_score(X, y_pred)
    db = davies_bouldin_score(X, y_pred)

    # External metrics (need partial labels)
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    pur = purity_score(y_true, y_pred)

    return sil, ch, db, ari, nmi, pur


def print_metrics(title, sil, ch, db, ari, nmi, pur):
    print(f"\n=== {title} ===")
    print(f"Silhouette Score        = {sil:.4f}")
    print(f"Calinski-Harabasz Index = {ch:.2f}")
    print(f"Davies-Bouldin Index    = {db:.4f}")
    print(f"ARI                     = {ari:.4f}")
    print(f"NMI                     = {nmi:.4f}")
    print(f"Purity                  = {pur:.4f}")


def main():
    # True labels
    genres = np.load(GENRES_PATH)  # (N,)

    # -------- Baseline input: audio MFCC + lyrics emb --------
    audio = np.load(AUDIO_PATH)    # (N, 40, 1300)
    lyrics = np.load(LYRICS_PATH)  # (N, 384)
    N = audio.shape[0]

    audio_flat = audio.reshape(N, -1)  # (N, 52000)
    X_raw = np.concatenate([audio_flat, lyrics], axis=1).astype(np.float32)

    # Standardize before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # PCA baseline (reduce to same latent dim as VAE = 32)
    pca = PCA(n_components=32, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    km_base = KMeans(n_clusters=K, random_state=42, n_init=10)
    clusters_pca = km_base.fit_predict(X_pca)

    sil, ch, db, ari, nmi, pur = clustering_metrics(X_pca, genres, clusters_pca)
    print_metrics("Baseline: PCA(32) + KMeans(10)", sil, ch, db, ari, nmi, pur)

    # -------- Your method input: VAE latent Z --------
    Z = np.load(Z_PATH)  # (N, 32)

    km_vae = KMeans(n_clusters=K, random_state=42, n_init=10)
    clusters_vae = km_vae.fit_predict(Z)

    sil2, ch2, db2, ari2, nmi2, pur2 = clustering_metrics(Z, genres, clusters_vae)
    print_metrics("Proposed: Beta-VAE Latent(32) + KMeans(10)", sil2, ch2, db2, ari2, nmi2, pur2)

    # Save cluster labels
    np.save("data/cluster_labels_pca.npy", clusters_pca)
    np.save("data/cluster_labels_vae.npy", clusters_vae)
    print("\nSaved:")
    print("  data/cluster_labels_pca.npy")
    print("  data/cluster_labels_vae.npy")


if __name__ == "__main__":
    main()
