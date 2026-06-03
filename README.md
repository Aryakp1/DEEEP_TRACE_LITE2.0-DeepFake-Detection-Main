# DeepTraceLite: Spatiotemporal Deepfake Forensic Framework

DeepTraceLite is an edge-optimized, industrial-grade **Spatiotemporal Deep Learning Framework** designed to detect digital video facial manipulations. By integrating highly compact spatial feature maps with temporal sequence learning, the framework isolates high-frequency facial anomalies (such as facial blending boundaries and edge artifacts) and tracks chronological discontinuities (such as flickering or unnatural blinking) across a video timeline.  

The system scales down parameter complexity to ensure operational stability on resource-constrained deployment environments, making it a viable candidate for mobile forensics and edge applications.  

---

## 📌 Core Engineering Pipeline

The architectural workflow is systematically structured into five distinct deployment phases:

1. **Deterministic Data Engineering:** Ingests a balanced dataset from the FaceForensics++ database, implementing strict reproducibility via locked pseudo-random number seeds ($42$) and automated persistence tracking using serialized JSON path configurations.  
2. **Local-First Caching & Sparse Extraction:** Bypasses cloud-link latency by building a virtual storage disk cache. It samples indices uniformly across the timeline via a linear spacing function (`np.linspace`) to scale continuous video streams down to a dense chronological sequence.  
3. **Targeted Facial Boundary Isolation:** Integrates Multi-task Cascaded Convolutional Networks (**MTCNN**) using a fast template shortcut—detecting facial landmarks on a central reference frame and locking that window across the clip to isolate the region of interest (ROI) and eliminate background environmental noise.  
4. **Hybrid Neural Processing (TinyCNN + BiGRU):** Condenses face crops through a custom, highly efficient 4-stage strided 2D Convolutional pipeline (**TinyCNN**) to generate dense spatial embeddings. These vectors are fed into a **Bidirectional GRU** equipped with **Global Average Temporal Pooling** ($v = o.mean(dim=1)$) to prevent localized anomaly dilution and evaluate chronological consistency uniformly.  
5. **Explainable AI (XAI) Verification:** Deploys **Integrated Gradients (IG)** to backtrace model predictions directly to pixel importance matrices. It applies an embedded CUDA fix (temporarily activating deterministic training parameters to let gradients bypass cuDNN recurrent layer restrictions) to yield clear, color-mapped visual attribution heatmaps supporting forensic audits.  

---

## 🛠️ Tech Stack & Advanced MLOps

* **Core Backend:** PyTorch (v2.x+) with Automatic Mixed Precision (AMP) training execution.  
* **Computer Vision Suite:** OpenCV (`opencv-python-headless`), `facenet-pytorch` (MTCNN module).  
* **Explainability Library:** Captum (Integrated Gradients).  
* **Industrial Optimization:** AdamW weight decay, OneCycleLR super-convergence scheduler, and gradient clipping constraints.  
* **Model Cross-Compilation:** TorchScript serialization and cross-platform ONNX graph primitives.  

---

## 📂 Project Architecture

```text
├── networks/
│   ├── __init__.py
│   ├── tiny_cnn.py          # Custom 4-stage strided conv feature extractor
│   └── tdam_module.py       # Temporal Difference Attention Module block
├── notebooks/
│   └── deeptacelite_main.ipynb # Full pipeline execution notebook
├── .gitignore               # Excludes massive binaries and raw datasets
└── README.md                # System documentation

# Expected data architecture inside local workspace / Cloud cache (Ignored by Git)
└── DeepTraceLite/
    ├── manifests/           # Cryptographic MD5 checksum digital receipts
    ├── splits/              # Stratified 80/10/10 split ledger JSON records
    ├── checkpoints/         # Serialized network weights (.pt files)
    └── exports/             # Cross-compiled production formats (ONNX, TorchScript)
