import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

Z = np.load("data/latent_z.npy")
genres = np.load("data/genres.npy")
clusters = np.load("data/cluster_labels.npy")

genre_names = sorted(list(set(genres)))
genre_to_id = {g: i for i, g in enumerate(genre_names)}
genre_ids = np.array([genre_to_id[g] for g in genres])

# ---------- t-SNE ----------
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
Z_tsne = tsne.fit_transform(Z)

plt.figure(figsize=(8, 6))
plt.scatter(Z_tsne[:, 0], Z_tsne[:, 1], c=genre_ids, cmap="tab10", s=10)
plt.colorbar(ticks=range(len(genre_names)), label="Genre")
plt.title("t-SNE Visualization of VAE Latent Space (Colored by Genre)")
plt.tight_layout()
plt.savefig("results/plots/tsne_genre.png")
plt.close()

# ---------- UMAP ----------
reducer = umap.UMAP(n_components=2, random_state=42)
Z_umap = reducer.fit_transform(Z)

plt.figure(figsize=(8, 6))
plt.scatter(Z_umap[:, 0], Z_umap[:, 1], c=clusters, cmap="tab10", s=10)
plt.colorbar(label="Cluster ID")
plt.title("UMAP Visualization of VAE Latent Space (Colored by Cluster)")
plt.tight_layout()
plt.savefig("results/plots/umap_cluster.png")
plt.close()

print("Saved visualizations to results/plots/")
