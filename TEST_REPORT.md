# TEST REPORT

Date: 2026-03-01
Repository: `Animals-Classification-on-Animals-10-Dataset-using-VGG16`

## 1. System Overview

Codebase components validated:
- Training pipeline: `train_zoo.py`
- Inference/UI pipeline: `app.py` (Streamlit)
- Model artifact: `zoo_bundle.pth`
- Dependencies: `requirements.txt`
- Tests: `tests/`

Observed runtime flow:
1. Train with `python train_zoo.py`.
2. Save artifact bundle (`model_state`, `class_names`, `metrics`) to `zoo_bundle.pth`.
3. Run app with `streamlit run app.py`.
4. App loads bundle and supports inference + diagnostics.

## 2. Issues Found

Issues were identified during audit and categorized as Critical / Major / Minor.

### Critical
- Inference preprocessing mismatch risk between train and app path due to manual app preprocessing and implicit resize behavior.
- AMP device handling in training path was CUDA-specific and not explicitly gated for CPU-safe behavior.

### Major
- Model loading lacked explicit `weights_only=True`.
- Incorrect Streamlit cache decorator use for returning `DataLoader` object.
- Dead/unused variables and imports in app pipeline.
- Requirements contained duplicate/unused entries and a CUDA-specific index URL.

### Minor
- Bare `except` in app gallery path.
- Redundant/unreachable logic in training test loop.
- Legacy/dead project files retained.
- Streamlit deprecated `use_container_width` usage warnings on image calls.

All listed items above were addressed in Phase 5 (see sections 5 and 6).

## 3. Tests Created

Test suite created under `tests/`:
- `tests/conftest.py`
- `tests/test_unit_train_utils.py`
- `tests/test_integration_training_pipeline.py`
- `tests/test_integration_inference_and_ml.py`
- `tests/test_edge_cases.py`

Coverage intent implemented:
- Unit tests:
  - `TransformedSubset` behavior.
  - Missing dataset handling in training entrypoint.
- Integration tests:
  - End-to-end training artifact creation and artifact schema validation.
  - App diagnostics test-split reconstruction.
- ML tests:
  - Model load path behavior.
  - Output shape validation for inference.
- Edge cases:
  - Missing model file handling.
  - Corrupted model load handling.

Observed test execution result:
- Command: `python -m pytest -q`
- Result: `7 passed`

## 4. Stress Results

Validation loop stress execution was run with system, ML, and data scenarios.

Observed aggregate:
- Total scenarios: `12`
- Failures: `0`

Observed scenario outcomes:
- `system_large_image_repeated_inference`: PASS (`prep_time=0.229s`, `infer_50=6.643s`)
- `system_rapid_streamlit_interactions`: PASS (`http_200=60`, `request_window=0.391s`)
- `ml_batch_processing`: PASS (`output_shape=(32, 10)`)
- `ml_cpu_gpu_switching`: PASS (`cpu_ok=True`, `gpu_status=True`)
- `ml_missing_model_file`: PASS (expected `FileNotFoundError`)
- `ml_corrupted_weights`: PASS (expected `UnpicklingError`)
- `data_invalid_values`: PASS (expected `UnidentifiedImageError`)
- `data_wrong_schema`: PASS (expected `FileNotFoundError`)
- `data_large_dataset`: PASS (`samples=1000`, `index_time=0.004s`)
- Non-image payload checks (CSV/video-like): PASS (rejected by PIL)

## 5. Fixes Applied

Evidence-based code fixes applied:

### `train_zoo.py`
- Added CPU-safe AMP gating (`amp_enabled`) and fallback context (`nullcontext`).
- Made resize interpolation explicit (`InterpolationMode.BILINEAR`) in train/eval transforms.
- Updated checkpoint load to explicit `weights_only=True`.
- Added explicit error when expected best temp checkpoint is missing.
- Removed unreachable empty-batch guard in test loop.

### `app.py`
- Removed unused import (`torch.nn as nn`).
- Set checkpoint load to explicit `weights_only=True`.
- Changed `load_test_data` cache decorator to `@st.cache_resource`.
- Replaced manual inference preprocessing with torchvision transform pipeline matching training normalization/resize settings.
- Removed dead diagnostics variable binding (`dataset_classes` -> `_`).
- Replaced bare `except` with `except Exception`.
- Replaced deprecated image `use_container_width` usage with `width="stretch"`.

### `requirements.txt`
- Removed CUDA-specific extra index URL.
- Removed duplicate dependency entry (`scikit-learn`).
- Removed unused dependencies (`albumentations`, `matplotlib`).

## 6. Cleanup Done

Removed dead or redundant files:
- `templates/index.html`
- `templates/result.html`
- `tree.txt`
- `.gitattributes`

Remaining `templates/` directory is empty.

## 7. Final Stability

Final verification evidence:
- Automated tests: `7 passed`.
- Stress validation loop: `12 scenarios`, `0 failures`.
- No regression observed in test/stress outputs after fixes.

Final status: **STABLE** based on executed tests and stress scenarios in this workspace.
