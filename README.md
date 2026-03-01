# ZooStation AI

## 1. Project Overview

This repository implements an image-classification system for 10 animal classes using PyTorch and Streamlit.

Implemented capabilities (from code):
- Train a `convnextv2_tiny` classifier on `raw-img/` (`train_zoo.py`).
- Save a model bundle (`zoo_bundle.pth`) containing model weights, class names, and test metrics.
- Run a Streamlit UI for:
   - single-image prediction,
   - Grad-CAM visualization,
   - diagnostics on a reconstructed test split,
   - dataset distribution/gallery views.

Problem addressed by implementation:
- Train and serve a local animal-image classifier with an interactive UI and basic model diagnostics.

---

## 2. Architecture Overview

### Components

| Component | File | Responsibility |
|---|---|---|
| UI Layer | `app.py` | Streamlit app, model loading, prediction, diagnostics, EDA views |
| Training/Backend Logic | `train_zoo.py` | Data loading/splitting, augmentation, training loop, evaluation, artifact creation |
| Test Suite | `tests/*.py` | Unit, integration, and edge-case validation of current behavior |

### `backend.py` status

`backend.py` is not present in this repository.
Backend training/inference logic is implemented directly in `train_zoo.py` and `app.py`.

### Workflow / agent system status

No workflow engine or agent framework is implemented in the codebase.

---

## 3. System Flow (Mandatory)

### Inference flow (UI path)
1. Streamlit starts and calls `load_zoo_model()`.
2. `load_zoo_model()` builds `convnextv2_tiny` and loads `zoo_bundle.pth`.
3. User uploads image in Identification tab.
4. Image is transformed (resize, tensor conversion, normalization).
5. Model performs forward pass and softmax probability computation.
6. UI shows predicted class and confidence.
7. Optional Grad-CAM path computes and displays attention map.

### Diagnostics flow (UI path)
1. User clicks “Run Full System Diagnostics”.
2. `load_test_data()` reconstructs the deterministic test split from `raw-img/`.
3. App runs batch inference on reconstructed test data.
4. App renders confusion matrix and classification report.

### Training flow
1. `train_zoo.py` loads `raw-img/` with `ImageFolder`.
2. Dataset is split 70/15/15 with seed `42`.
3. Model trains with augmentation + validation + early stopping.
4. Best checkpoint is evaluated on test split.
5. `zoo_bundle.pth` is saved with `model_state`, `class_names`, `metrics`.

```mermaid
flowchart TD
      A[User runs streamlit run app.py] --> B[load_zoo_model]
      B --> C{zoo_bundle.pth available?}
      C -- No --> D[Show error in UI]
      C -- Yes --> E[Build convnextv2_tiny and load weights]
      E --> F[User uploads image]
      F --> G[Resize + ToTensor + Normalize]
      G --> H[Model inference + softmax]
      H --> I[Show predicted class + confidence]
      I --> J[Optional Grad-CAM visualization]

      E --> K[User clicks diagnostics]
      K --> L[Rebuild deterministic test split]
      L --> M[Batch inference on test loader]
      M --> N[Confusion matrix + classification report]

      O[User runs python train_zoo.py] --> P[ImageFolder(raw-img)]
      P --> Q[Split 70/15/15 seed=42]
      Q --> R[Train/Val/Test DataLoaders]
      R --> S[Train + validate + early stopping]
      S --> T[Load best temp checkpoint]
      T --> U[Test evaluation]
      U --> V[Save zoo_bundle.pth]
```

---

## 4. Workflow / Agent Logic

No agent runtime or node-based workflow system is implemented.

Conditional logic present in app/training code:
- Model-file existence handling in `load_zoo_model()`.
- Dataset existence handling in training and diagnostics paths.
- Early stopping in training loop based on validation loss.
- Conditional diagnostics execution on button click.

---

## 5. Data Model / State Structure

### Model artifact (`zoo_bundle.pth`)
Saved dictionary keys:
- `model_state`: PyTorch state dict for model parameters.
- `class_names`: ordered class-name list used for display.
- `metrics`: dictionary with:
   - `accuracy`
   - `f1_score`

### Runtime structures (implemented)
- Split lengths computed from dataset size:
   - `train_len = int(0.7 * total_len)`
   - `val_len = int(0.15 * total_len)`
   - `test_len = total_len - train_len - val_len`
- Prediction outputs:
   - logits tensor from model,
   - softmax probabilities,
   - predicted index and confidence.

---

## 6. Core Modules Breakdown

### `train_zoo.py`

| Function / Class | Input | Output | Behavior |
|---|---|---|---|
| `TransformedSubset` | subset, transform | dataset-like wrapper | Applies transform at item access time |
| `main()` | none (uses module config constants) | none (side effects) | Trains model, evaluates, saves `zoo_bundle.pth` |

Key behavior in `main()`:
- Builds train/eval transforms.
- Loads and splits data with fixed seed.
- Trains `convnextv2_tiny` using `CrossEntropyLoss` + `AdamW`.
- Uses AMP only when CUDA is active.
- Performs early stopping on validation loss.
- Evaluates on test split and saves artifact.

### `app.py`

| Function / Class | Input | Output | Behavior |
|---|---|---|---|
| `TransformedSubset` | subset, transform | dataset-like wrapper | Applies transform for diagnostics test set |
| `load_zoo_model()` | none | `(model, class_names)` or `(None, None)` | Builds model and loads checkpoint |
| `load_test_data()` | none | `(test_loader, classes)` or `(None, None)` | Reconstructs deterministic test split |

Tab behaviors:
- **Identification**: file upload, preprocessing, inference, confidence display, optional Grad-CAM.
- **Intelligence**: on button click, runs test-set inference and renders confusion matrix/report.
- **Archive**: scans dataset folders, shows class counts and sample gallery.
- **Neural Architecture**: static descriptive markdown section.

---

## 7. Security Model

Implemented protections/constraints in code:
- File upload limits by extension in UI uploader: `jpg`, `png`, `jpeg`.
- Inference and diagnostics are local model execution only (no external API calls in app/training code).
- Exception handling for model-load failure and Grad-CAM failure paths.

Not implemented:
- No explicit file-size validation for uploads.
- No content-type or malware scanning.
- No authentication/authorization layer.

---

## 8. LLM / Provider Integration

No LLM provider integration is implemented in this repository.
No model-provider selection or fallback logic exists.

---

## 9. Setup & Installation

### 1) Create and activate virtual environment

Windows (PowerShell):
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Prepare dataset
Place the dataset folder as:
```text
raw-img/
   cane/
   cavallo/
   elefante/
   farfalla/
   gallina/
   gatto/
   mucca/
   pecora/
   ragno/
   scoiattolo/
```

---

## 10. Running the Application

### Train model artifact
```bash
python train_zoo.py
```

Expected behavior:
- Training/validation progress prints to console.
- Final test metrics print to console.
- `zoo_bundle.pth` is generated in project root.

### Launch Streamlit UI
```bash
streamlit run app.py
```

Expected behavior:
- App opens with four tabs: Identification, Intelligence, Archive, Neural Architecture.
- If `zoo_bundle.pth` is missing, app displays an error message.

---

## 11. Testing

Framework:
- `pytest`

Test modules present:
- `tests/test_unit_train_utils.py`
- `tests/test_integration_training_pipeline.py`
- `tests/test_integration_inference_and_ml.py`
- `tests/test_edge_cases.py`

Run tests:
```bash
python -m pytest -q
```

Covered behavior (from tests):
- Subset transform behavior.
- Missing dataset handling in training entrypoint.
- Training artifact structure and metric key presence.
- Test-split reconstruction behavior.
- Model load success path (mocked checkpoint load).
- Missing/corrupted model-load handling paths.

---

## 12. Limitations

Code-observable constraints:
- `app.py` executes model loading at module import time; this couples import and runtime side effects.
- Diagnostics path assumes usable loaded model when button is pressed.
- Dependencies are not version-pinned in `requirements.txt`, so installs can vary over time.
- `backend.py` is absent; backend responsibilities are split between `app.py` and `train_zoo.py`.
- No built-in dataset download utility; dataset placement is manual.

---

## 13. Future Improvements (Grounded)

Realistic improvements implied by current implementation:
- Add explicit diagnostics guard for unavailable model before inference loop.
- Add stronger tests for real checkpoint load path (non-mocked file path test).
- Add metric value-range assertions in training integration tests.
- Pin dependency versions for reproducible environments.
