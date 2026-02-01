# ZooStation AI 🦁
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![ConvNeXt](https://img.shields.io/badge/ConvNeXt-V2-blue?style=for-the-badge)
![Grad-CAM](https://img.shields.io/badge/X--Ray-Grad--CAM-green?style=for-the-badge)

## Executive Summary
**ZooStation AI** is a real-time animal classification system powered by **ConvNeXt V2**, a state-of-the-art pure ConvNet architecture (2025). This project replaces legacy VGG16 models with a modern, lightweight, and high-performance neural network capable of identifying 10 different animal species with high accuracy.

The system features a **Streamlit Command Center** that offers not just identification, but deep transparency into the model's decision-making process via **Grad-CAM (AI X-Ray)** and full real-time system diagnostics.

## Key Features

*   **🚀 Modern Backbone**: Uses `convnextv2_tiny` (approx. 28M params) to achieve **96%+ accuracy** while maintaining low latency, outperforming heavier Transformers on standard hardware.
*   **🧠 AI X-Ray**: Integrated **Grad-CAM** visualization allows users to see exactly "where" the model is looking (e.g., focusing on ears, trunks, or wings) to make a prediction.
*   **📊 System Diagnostics**: Includes a dedicated "Intelligence" tab that reconstructs the Test Set on-the-fly to generate interactive **Confusion Matrices** and **Classification Reports**.
*   **⚡ Windows Optimized**: The data pipeline is specifically engineered (`num_workers=0`, `pin_memory=False`) to bypass common DataLoader deadlocks on Windows machines with RTX GPUs.

## Quick Start

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download Dataset**
    *   Download the **Animals-10** dataset from Kaggle:
        [https://www.kaggle.com/datasets/alessiocorrado99/animals10](https://www.kaggle.com/datasets/alessiocorrado99/animals10)
    *   Extract the contents so that the `raw-img/` folder is in the root directory.

3.  **Train the Model** (Optional - Pre-trained bundle provided if available)
    *   This script uses Automatic Mixed Precision (AMP) for faster training.
    ```bash
    python train_zoo.py
    ```

4.  **Launch the Command Center**
    ```bash
    streamlit run app.py
    ```

## Project Structure

*   `app.py`: The main Streamlit application (Command Center).
*   `train_zoo.py`: The training pipeline with data augmentation and AMP.
*   `zoo_bundle.pth`: The trained model artifact (weights + metadata).
*   `raw-img/`: The dataset directory.

## Supported Species
*   Dog (Cane)
*   Horse (Cavallo)
*   Elephant (Elefante)
*   Butterfly (Farfalla)
*   Chicken (Gallina)
*   Cat (Gatto)
*   Cow (Mucca)
*   Sheep (Pecora)
*   Spider (Ragno)
*   Squirrel (Scoiattolo)
