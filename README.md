# Multimodal Music Genre Clustering using Beta-VAE

## Overview
This project implements a multimodal clustering pipeline using audio (MFCC) and lyrics embeddings.
A Beta-VAE is trained to learn latent representations, followed by K-Means clustering.

## Dataset
- GTZAN Music Genre Dataset
- Audio: MFCC features
- Lyrics: Whisper transcription + SentenceTransformer embeddings

## Pipeline
1. Audio feature extraction (MFCC)
2. Lyrics transcription and embedding
3. Multimodal Beta-VAE training
4. K-Means clustering in latent space
5. Visualization using t-SNE and UMAP
6. Evaluation using ARI, NMI, Purity, Silhouette, CH, DB index

## How to Run
```bash
pip install -r requirements.txt
python src/features.py
python src/vae.py
python src/clustering.py
python src/visualize.py

Results

ARI: 0.134

NMI: 0.249

Purity: 0.387
