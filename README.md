
# 🍎 Fruit-306: A Large-Scale Multimodal Fruit Dataset & LLM-Enhanced Ensemble Framework

<div align="center">

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)]()
[![Project Page](https://img.shields.io/badge/Project-Page-orange.svg)](https://mybkgjvgnd.github.io/Fruit-306-Dataset/)

**A new benchmark for fine-grained fruit recognition, long-tail learning, and multimodal understanding with LLM arbitration.**

[📄 Anonymous Paper & Project Page](https://mybkgjvgnd.github.io/Fruit-306-Dataset/) | [📥 Download Data](#-download--preparation)

</div>

---

## 📖 Overview

**Fruit-306** is a comprehensive, large-scale dataset designed to advance research in **Fine-Grained Visual Categorization (FGVC)** and **Multimodal Learning** within the agricultural domain. 

Unlike existing datasets that suffer from limited classes, synthetic backgrounds, or lack of semantic annotations, Fruit-306 offers:

*   🌟 **306 Fine-Grained Classes**: Covering a wide variety of fruits with distinct species and varieties.
*   🌍 **Real-World Complexity**: Images captured in diverse environments with varying lighting, occlusions, and complex backgrounds.
*   📉 **Long-Tail Distribution**: Naturally reflects real-world data scarcity, ideal for researching imbalanced learning strategies.
*   📝 **Multimodal Annotations**: Includes high-quality images paired with rich text descriptions (sourced from agricultural databases).
*   🎯 **Pixel-Level Segmentation**: Provides segmentation masks for precise object localization.
*   🤖 **LLM-Enhanced Baseline**: We provide a novel ensemble framework leveraging Large Language Models (**Qwen-VL**) as an arbitrator to resolve conflicts in hard samples.

---

## 📊 Dataset Statistics

| Feature | Details | Description |
| :--- | :--- | :--- |
| **Total Classes** | **306** | Distinct fruit species/varieties |
| **Total Images** | **~116k** | High-resolution real-world images |
| **Modalities** | Image, Text, Mask | Visual + Semantic + Spatial |
| **Distribution** | Long-Tail | Significant class imbalance |
| **Quality Grades** | 3 Levels | High-quality, Defective, Flawed |

> **Comparison**: Fruit-306 significantly outperforms existing datasets like *Fruits-360* (limited classes, clean background) and *VegFru* (lack of text/segmentation) in terms of scale, complexity, and modality richness.

---

## 🖼️ Sample Visualization

Below are representative examples from the Fruit-306 dataset, showcasing the diversity across **different growth stages**, **environmental conditions**, and **fine-grained categories**.

<div align="center">
  <table>
    <tr>
      <td colspan="3" align="center"><b>🍎 Apple Varieties - Fine-Grained Differences</b></td>
    </tr>
    <tr>
      <td align="center">
        <img src="assets/01.jpg" width="180" alt="Red Fuji Apple"/>
        <br><i>cloudberry</i>
      </td>
      <td align="center">
        <img src="assets/02.jpg" width="180" alt="Green Apple"/>
        <br><i>blueberry</i>
      </td>
      <td align="center">
        <img src="assets/03.jpg" width="180" alt="Snake Apple"/>
        <br><i>dewberry</i>
      </td>
    </tr>
    <tr>
      <td colspan="3" align="center"><b>🍊 Citrus Varieties - Shape & Color Variations</b></td>
    </tr>
    <tr>
      <td align="center">
        <img src="assets/04.jpg" width="180" alt="Navel Orange"/>
        <br><i>sweet_orange</i>
      </td>
      <td align="center">
        <img src="assets/05.jpg" width="180" alt="Green Orange"/>
        <br><i>lime</i>
      </td>
      <td align="center">
        <img src="assets/06.jpg" width="180" alt="Pomelo"/>
        <br><i>green_orange</i>
      </td>
    </tr>
    <tr>
      <td colspan="3" align="center"><b>🌱 Different Growth Stages & Scenes</b></td>
    </tr>
    <tr>
      <td align="center">
        <img src="assets/07.jpg" width="180" alt="Unripe Strawberry"/>
        <br><i>Strawberry (Unripe)</i>
      </td>
      <td align="center">
        <img src="assets/08.jpg" width="180" alt="Ripe Strawberry"/>
        <br><i>Strawberry (Ripe)</i>
      </td>
      <td align="center">
        <img src="assets/09.jpg" width="180" alt="Strawberry Harvest Scene"/>
        <br><i>Strawberry (Harvest Scene)</i>
      </td>
    </tr>
  </table>
</div>

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="assets/class_distribution.png" alt="Class Distribution" width="400"/>
        <br><i>Figure 2: Long-tail distribution across 306 fruit classes</i>
      </td>
      <td align="center">
        <img src="assets/growth_stage_dist.png" alt="Growth Stage Distribution" width="400"/>
        <br><i>Figure 3: Distribution of different growth stages in the dataset</i>
      </td>
    </tr>
  </table>
</div>

> **Key Features Demonstrated:**
> - **Fine-grained distinctions** between visually similar varieties (e.g., different apple types)
> - **Temporal diversity** capturing fruits from unripe to ripe stages
> - **Environmental variability** including natural occlusions, complex orchard backgrounds, and lighting changes
> - **Real-world conditions** that challenge traditional classification models

*(Note: Please replace the example image paths with your actual dataset samples. Recommended to include 2-3 representative images per fruit category showing different conditions.)*

## 📥 Download & Preparation

Due to the large size of the dataset (~23GB+), the image files are hosted externally. Please download them using the links below and organize the directory structure as shown.

### 1. Download Links

| Source | Link | Notes |
| :--- | :--- | :--- |
| **Baidu Netdisk** | [点击下载](https://pan.baidu.com/s/18K6sAzKoBi-LBsXQcYLg6w?pwd=th9a) (提取码: `th9a`) | For users in China |

### 2. Directory Structure

After downloading, please organize your local directory as follows:

```text
After downloading, please organize your local directory as follows:

```text
Fruit-306/
├── train/                # Training images organized by class folders
│   ├── class_001/        # Fruit class 1 folder
│   ├── class_002/        # Fruit class 2 folder
│   └── ...                # Other fruit class folders (total 306 classes)
├── val/                  # Validation images organized by class folders
│   ├── class_001/
│   ├── class_002/
│   └── ...
├── test/                 # Test images organized by class folders
│   ├── class_001/
│   ├── class_002/
│   └── ...
├── fruit_name_mapping.txt # Mapping of class IDs to fruit names
└── dataset_info.txt       # Dataset statistics and metadata
