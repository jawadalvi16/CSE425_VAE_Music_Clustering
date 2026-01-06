# Multimodal Music Genre Clustering using Beta-VAE

## Overview
This repository contains the implementation of a **multimodal unsupervised music genre clustering project**, developed as part of the **CSE425 (Neural Networks)** course.

The objective of this project is to learn **compact and disentangled latent representations** of music tracks using a **Beta-Variational Autoencoder (β-VAE)** and perform clustering in the learned latent space. The project explores clustering performance using **audio features** and auxiliary **lyrical information**, and compares VAE based representations with classical baselines.

Audio features are extracted using **Mel-Frequency Cepstral Coeffal Coefficients (MFCCs)**. Lyrics are generated via automatic speech recognition and embedded using a pretrained language model. Clustering is performed using **K-Means**, and results are evaluated using multiple unsupervised and label based metrics.

---

## Dataset
- **GTZAN Music Genre Dataset**
- 10 genres, 100 tracks per genre
- Audio duration: ~30 seconds per track
- Audio features: MFCC (40 coefficients)
- Lyrics: Whisper transcription + SentenceTransformer embeddings

> **Note:** Due to size constraints, the dataset is not included in this repository. We have to download the GTZAN dataset separately and place it according to the expected directory structure.

---

## Methodology / Pipeline
1. Audio feature extraction using MFCC  
2. Lyrics transcription using Whisper  
3. Lyrics embedding using SentenceTransformer  
4. Learning latent representations using **Beta-VAE**  
5. K-Means clustering in the latent space  
6. Evaluation using Silhouette Score, CH Index, DB Index, ARI, NMI, and Purity  
7. Visualization using t-SNE and UMAP  

---

## Baseline Method
A classical baseline is implemented for comparison:
- **PCA (32 dimensions) + K-Means (10 clusters)**

This baseline enables comparison between linear dimensionality reduction and nonlinear representation learning using the Beta-VAE.

---

## Repository Structure and File Description
Although a specific file structure and task-wise notebooks were suggested, this project follows a **custom but well-documented structure**. All tasks (Easy, Medium, and Hard) are implemented across the notebooks and source files described below.
```
project/
│
├── data/
│   ├── audio/
│   └── lyrics/
│
├── notebooks/
│   ├── exploratory and experiment notebooks
│   └── feature extraction, VAE training, and clustering analysis
│
├── src/
│   ├── features.py      # Audio and lyric feature extraction
│   ├── vae.py           # Beta-VAE model and training pipeline
│   ├── clustering.py    # K-Means clustering in latent space
│   ├── evaluation.py    # Clustering metrics (ARI, NMI, Purity, etc.)
│   └── visualize.py     # t-SNE and UMAP visualizations
│
├── results/
│   ├── latent_visualizations/
│   └── clustering_metrics.csv
│
├── README.md
└── requirements.txt
```

### Task Mapping
- **Easy Task:** Basic VAE implementation, K-Means clustering, baseline PCA + K-Means, and basic visualizations  
- **Medium Task:** Enhanced VAE architecture, inclusion of lyric embeddings, multiple clustering metrics  
- **Hard Task:** Beta-VAE, multi-modal feature analysis, advanced evaluation metrics (ARI, NMI, Purity), and detailed latent space visualizations  

Tasks are **not separated into individual easy/medium/hard notebooks**, but are **clearly implemented and documented** within the codebase and experiments.

---

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
python src/features.py
python src/vae.py
python src/clustering.py
python src/visualize.py
```

ARI   : 0.134
NMI   : 0.249
Purity: 0.387


---

## References

~~~
Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes.

Higgins, I. et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.

Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals.

Radford, A. et al. (2023). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision.

Reimers, N., & Gurevych, I. (2019). Sentence-BERT.
~~~




