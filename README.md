
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

Below are representative examples from the Fruit-306 dataset, showcasing the diversity in categories, quality grades (High-quality, Defective, Flawed), and environmental conditions.

<div align="center">
  <table>
    <tr>
      <td align="center"><b>High-Quality</b><br><i>(Clear features)</i></td>
      <td align="center"><b>Defective</b><br><i>(Minor bruises/scars)</i></td>
    </tr>
    <tr>
      <td align="center">
        <img src="assets/木瓜_0258.jpeg" width="200" alt="High Quality Example"/>
        <br><i>Papaya</i>
      </td>
      <td align="center">
        <img src="assets/蔓越莓_01.jpg" width="200" alt="High Quality Example"/>
        <br><i>蔓越莓</i>
      </td>
    </tr>
    <tr>
      <td align="center">
        <img src="assets/青橙_0146.jpg" width="200" alt="High Quality Example 2"/>
        <br><i>Green Orange</i>
      </td>
      <td align="center">
        <img src="assets/青苹果_0442.jpg" width="200" alt="High Quality Example 2"/>
        <br><i>Green Apple</i>
      </td>
      </td>
    </tr>
  </table>
  <p><i>Figure 1: Examples across different quality grades and categories. The dataset includes complex backgrounds and varying lighting conditions.</i></p>
</div>

<div align="center">
  <img src="assets/long_tail_dist.png" alt="Long-Tail Distribution Chart" width="600"/>
  <p><i>Figure 2: The long-tail distribution of the 306 fruit classes.</i></p>
</div>

*(Note: Please ensure you place your sample images in the `assets/` folder and update the filenames in the table above if necessary.)*
## 📥 Download & Preparation

Due to the large size of the dataset (~50GB+), the image files are hosted externally. Please download them using the links below and organize the directory structure as shown.

### 1. Download Links

| Source | Link | Notes |
| :--- | :--- | :--- |
| **Hugging Face** | [Coming Soon / Link] | Recommended for programmatic access |
| **Google Drive** | [Link] | Full dataset archive |
| **Baidu Netdisk** | [Link] (Code: `xxxx`) | For users in China |

### 2. Directory Structure

After downloading, please organize your local directory as follows:

```text
Fruit-306/
├── images/              # Extracted images organized by class or flat structure
│   ├── class_001/
│   ├── class_002/
│   └── ...
├── annotations/         # JSON files for BBox and Segmentation (COCO format)
│   ├── train.json
│   ├── val.json
│   └── test.json
├── texts/               # Text descriptions
│   └── descriptions.json
├── splits/              # Train/Val/Test split lists
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
└── class_names.txt      # Mapping of ID to Class Name
