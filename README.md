# Multimodal Music Genre Clustering using Beta-VAE

## Overview
This repository contains the implementation of a multimodal music genre clustering project developed as part of the **CSE425 (Machine Learning)** course. The goal of this project is to explore **unsupervised music genre clustering** by learning compact and structured latent representations from both **audio signals and lyrical content**.

Audio features are extracted using Mel-Frequency Cepstral Coefficients (MFCCs), while lyrics are automatically generated using a speech-to-text model and embedded using a pretrained language model. A **Beta-Variational Autoencoder (Beta-VAE)** is trained to learn disentangled latent representations, followed by clustering using **K-Means**.

---

## Dataset
- **GTZAN Music Genre Dataset**
- 10 music genres, 100 tracks per genre
- Audio duration: ~30 seconds per track
- Audio features: MFCC (40 coefficients)
- Lyrics: Whisper transcription + SentenceTransformer embeddings

> **Note:** Due to licensing and size constraints, the dataset is not included in this repository. Users are expected to obtain the GTZAN dataset separately.

---

## Pipeline
1. Audio feature extraction (MFCC)
2. Lyrics transcription using Whisper
3. Lyrics embedding using SentenceTransformer
4. Beta-VAE training on audio features
5. K-Means clustering in latent space
6. Evaluation using Silhouette, CH, DB, ARI, NMI, and Purity
7. Visualization using t-SNE and UMAP

---

## Baseline
For comparison, a classical baseline approach is implemented:
- **PCA (32 dimensions) + K-Means (10 clusters)**

This allows comparison between linear dimensionality reduction and nonlinear representation learning using Beta-VAE.

---

---

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt


Run the pipeline:
python src/features.py
python src/vae.py
python src/clustering.py
python src/visualize.py


Sample Results (Beta-VAE + K-Means)
ARI   : 0.134
NMI   : 0.249
Purity: 0.387

References

Kingma, D. P., and Welling, M. (2014). Auto-Encoding Variational Bayes.

Higgins, I. et al. (2017). beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.

Tzanetakis, G., and Cook, P. (2002). Musical genre classification of audio signals.

Radford, A. et al. (2023). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision.

Reimers, N., and Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.


