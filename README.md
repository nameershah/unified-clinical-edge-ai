# Unified Clinical Edge AI

A deterministic safety architecture for clinical tissue classification on edge devices, built on PathMNIST histopathology data. This project decouples probabilistic deep learning inference from deterministic execution policy using the H.E.A.R.L. (Hyper-Efficient Agentic Routing Layer) framework.

**Author:** Muhammad Nameer Shah  
**Affiliation:** University of Agriculture Peshawar, BS Artificial Intelligence  
**Contact:** smns3960@gmail.com  

---

## The Problem

Large language models and deep learning classifiers deployed in clinical settings share a fundamental flaw: they produce probabilistic outputs with no built-in mechanism to block low-confidence decisions from reaching downstream systems. In diagnostic AI, a hallucinated or low-confidence prediction that proceeds unchecked is not a software bug — it is a patient safety failure.

## The Architecture

This project proposes a two-layer clinical inference pipeline:

```
PathMNIST Patch
      |
      v
ResNet-18 Classifier  -->  Probability Tensor (9 classes)
      |
      v
H.E.A.R.L. Router  -->  Deterministic Gate
      |                        |
      |              [BLOCK if U(M) < threshold]
      v
Gemini RAG Report Generator  -->  Final Clinical Report
```

The H.E.A.R.L. utility function scores each model output before it is allowed to proceed:

```
U(M) = w1 * A(M) - w2 * C(M) - w3 * L(M)
```

Where:
- `A(M)` = Classification confidence (accuracy signal)
- `C(M)` = Computational cost of the inference pass
- `L(M)` = Latency of the inference pass
- `w1, w2, w3` = Tunable policy weights

If `U(M)` falls below the defined threshold, the output is blocked. No report is generated. No downstream system receives a low-confidence prediction.

---

## Phase 1 Results — ResNet-18 Baseline

| Metric | Value |
|---|---|
| Dataset | PathMNIST (89,996 train / 10,004 val / 7,180 test) |
| Model | ResNet-18 (ImageNet pretrained, full fine-tune) |
| Test Accuracy | 92.21% |
| Macro AUC | 0.9861 |
| Epochs | 15 |
| Optimizer | Adam + Cosine Annealing LR |

Training curves and confusion matrix are available in `exports/`.

---

## Repository Structure

```
unified-clinical-edge-ai/
|
|-- notebooks/
|   |-- unified_clinical_edge_ai_phase1_baseline.ipynb   # Phase 1: Training
|   |-- unified_clinical_edge_ai_phase2_hearl.ipynb      # Phase 2: Quantization + H.E.A.R.L.
|   |-- unified_clinical_edge_ai_phase3_rag.ipynb        # Phase 3: Gemini RAG integration
|
|-- exports/
|   |-- resnet18_pathmnist_best.pth                      # Best model weights
|   |-- checkpoint_full.pth                              # Full training checkpoint
|   |-- metrics.json                                     # Evaluation metrics
|   |-- training_curves.png                              # Loss and accuracy curves
|   |-- confusion_matrix.png                             # Per-class confusion matrix
|
|-- docs/
|   |-- technical_paper.pdf                              # 3-page academic preprint
|
|-- src/
|   |-- hearl_router.py                                  # H.E.A.R.L. deterministic kernel
|   |-- quantize.py                                      # Edge quantization pipeline
|   |-- rag_pipeline.py                                  # Gemini RAG clinical report generator
|
|-- README.md
|-- LICENSE
```

---

## Quickstart

```bash
git clone https://github.com/nameershah/unified-clinical-edge-ai.git
cd unified-clinical-edge-ai
pip install torch torchvision medmnist scikit-learn matplotlib seaborn
```

Open `notebooks/unified_clinical_edge_ai_phase1_baseline.ipynb` in Google Colab with a T4 GPU and run all cells.

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| 1 — Baseline Training | Done | ResNet-18 on PathMNIST, 92.21% accuracy |
| 2 — Edge Quantization + H.E.A.R.L. | In progress | Dynamic quantization + deterministic routing kernel |
| 3 — RAG Integration | Pending | Gemini API clinical report generation gated by H.E.A.R.L. |
| 4 — Technical Paper | Pending | 3-page academic preprint |

---

## Citation

If you use this work, please cite:

```
Shah, M. N. (2026). Unified Clinical Edge AI: Decoupling Probabilistic Inference 
from Deterministic Execution in Clinical Tissue Classification. 
GitHub. https://github.com/nameershah/unified-clinical-edge-ai
```

---

## License

This project is licensed under the Muhammad Nameer Shah Open Research License. See `LICENSE` for full terms.
