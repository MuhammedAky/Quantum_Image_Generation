# Quantum Image Generation

A research project investigating hybrid classical-quantum generative models for image representation and generation.

The project is currently being developed incrementally, beginning with classical Autoencoder and Variational Autoencoder baselines before introducing quantum components.

---

## Project Status

Current stage:

- [x] MNIST veri seti ile klasik AE eğitildi
- [x] Farklı latent boyutları denendi
- [x] Rekonstrüksiyon metrikleri kaydedildi
- [x] Klasik VAE eğitildi
- [x] VAE metrikleri kaydedildi
- [x] Hybrid VAE oluşturuldu
- [x] PennyLane simülatörü kullanıldı
- [x] Klasik latent bilgilerin kuantum katmana aktarılması denendi
- [x] Gerçek QPU seçild
- [x] Tek-qubit Ry devresi gerçek QPU’da çalıştırıldı
- [x] 0, π/4 ve π/2 açıları için ölçüm alındı
- [x] QPU sonuçları kaydedildi
- [x] İdeal ve deneysel sonuçlar karşılaştırıldı
---

## Current Objective

The first stage of the project focuses on understanding and implementing a classical Autoencoder.

The Autoencoder learns a compressed latent representation of MNIST images and attempts to reconstruct the original images from this representation.

The current pipeline is:

Input Image
    ↓
Encoder
    ↓
Latent Representation
    ↓
Decoder
    ↓
Reconstructed Image

For MNIST images:

28 × 28 = 784 input pixels

The current Autoencoder compresses the 784-dimensional input into an 8-dimensional latent representation.

---

## Model Architecture

### Encoder

784
↓
128
↓
8

### Decoder

8
↓
128
↓
784

The complete model is:

784 → 128 → 8 → 128 → 784

The latent representation is learned through neural network optimization.

---

## Dataset

The current implementation uses the MNIST handwritten digit dataset.

The dataset is downloaded automatically by the training script.

The MNIST labels are not used for the Autoencoder reconstruction task.

The model receives an image and attempts to reconstruct the same image.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/MuhammedAky/Quantum_Image_Generation