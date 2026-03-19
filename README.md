# 🍎 Fruit-306: A Large-Scale Multimodal Fruit Dataset & FruitEnsemble Framework

<!-- <div align="center">

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)]()
[![Project Page](https://img.shields.io/badge/Project-Page-orange.svg)](https://mybkgjvgnd.github.io/Fruit-306-Dataset/)

**A new benchmark for fine-grained fruit recognition with MLLM-guided ensemble arbitration.**

[📄 Anonymous Paper & Project Page](https://mybkgjvgnd.github.io/Fruit-306-Dataset/) | [📥 Download Data](#-download--preparation)

</div> -->

---

## 📖 Overview

**Fruit-306** is a comprehensive, large-scale dataset designed to advance research in **Fine-Grained Visual Categorization (FGVC)** within the agricultural domain. Unlike existing datasets that suffer from limited classes or controlled backgrounds, Fruit-306 offers:

*   🌟 **306 Fine-Grained Classes**: Covering a wide variety of fruit cultivars with distinct species and varieties
*   🌍 **Real-World Complexity**: 116,233 images captured in diverse environments with varying lighting, occlusions, and complex backgrounds
*   📉 **Long-Tail Distribution**: Imbalance ratio of 50:1 (max: 1,276 images for jaboticaba, min: 25 images for muscadine_grape)
*   📝 **Multimodal Annotations**: Images paired with expert-curated textual morphological descriptions
*   🤖 **FruitEnsemble Framework**: A novel dynamic two-stage framework using MLLM (Qwen-VL-Plus) as an arbiter for ambiguous samples

---
# 🍎 Fruit-306 Dataset: Complete English Variety List

This document lists all **306 fruit categories** included in the **Fruit-306 Dataset**. Every entry is classified as a **fruit** based on botanical definitions (developing from the ovary of a flower) or standard computer vision fruit recognition benchmarks.

## 📊 Dataset Overview
- **Total Classes**: 306 Fruit Varieties
- **Language**: English Labels (Snake_case format)
- **Scope**: Common fruits, exotic tropicals, berries, citrus, botanical vegetables (tomatoes/peppers), and nut-fruits.

---

## 🍇 Categorized Fruit List

### 1. Common & Global Fruits
Standard fleshy fruits found in markets worldwide.
- `apple` (variants: `red_apple`, `green_apple`, `delicious_apple`, `crab_apple`, `mountain_apple`, `malay_apple`, `java_apple`, `rose_apple`, `sugar_apple`, `custard_apple`, `star_apple`, `wood_apple`)
- `banana` (variants: `red_banana`, `thai_banana`, `plantain`)
- `orange` (variants: `sweet_orange`, `green_orange`, `wild_orange`, `bergamot_orange`)
- `grape` (variants: `red_grape_global`, `green_grapevine`, `black_grape`, `purple_grape`, `white_grape`, `baby_grape`, `seedless_grape`, `concord_grape`, `kyoho_grape`)
- `strawberry`, `blueberry`, `raspberry`, `blackberry`, `cranberry`, `mulberry`
- `watermelon`, `cantaloupe`, `honeydew_melon`, `muskmelon`
- `peach`, `nectarine`, `apricot`, `plum` (variants: `black_plum`, `golden_plum`, `greengage`)
- `cherry` (variants: `sweet_cherry`, `sour_cherry`, `surinam_cherry`)
- `fig` (variants: `desert_fig`, `indian_fig`, `purple_fig`)
- `kiwi` (variants: `gold_kiwi`, `green_kiwi`)
- `pear` (variants: `asian_pear`, `european_pear`, `green_pear`, `red_pear`)
- `pineapple`, `mango`, `papaya`, `guava`, `pomegranate`
- `lychee`, `longan`, `sapodilla`, `canistel`, `lucuma`, `cupuacu`, `camu_camu`
- `bael_fruit`, `baobab_fruit`, `noni_fruit`, `miracle_fruit`, `monk_fruit`, `pawpaw`
- `santol`, `marang`, `mamey_sapote`, `white_sapote`, `black_sapote`, `yellow_sapote`
- `nance`, `pequi`, `pupunha`, `ita_palm_fruit`, `oil_palm_fruit`
- `coconut` (variants: `gold_coconut`)
- `date` (variants: `medjool_date`, `black_date`, `red_date`, `purple_date`)
- `jujube` (variants: `red_jujube`, `green_jujube`, `yellow_jujube`, `wild_jujube`)
- `tamarind` (variants: `manila_tamarind`, `velvet_tamarind`)
- `persimmon` (variants: `flat_persimmon`, `snow_persimmon`, `yellow_persimmon`, `bicolor_persimmon`)

### 2. Exotic & Tropical Fruits
Rare and distinctive fruits primarily from tropical regions.
- `durian` (variants: `durian_montong`, `golden_pillow_durian`)
- `mangosteen`, `rambutan`, `salak` (Snake Fruit)
- `dragon_fruit` / `pitaya`, `prickly_pear`, `cactus_fruit`
- `jackfruit_small`, `breadfruit`
- `jaboticaba`, `wampee`, `langsat`, `longkong`
- `passion_fruit` (variants: `giant_granadilla`, `tamarillo`, `tree_tomato`)
- `soursop`, `cherimoya`, `atemoya`, `feijoa`
- `ackee`, `durian_variants`, `mangosteen_variants`

### 3. Citrus Family Fruits
All members of the Rutaceae family.
- `lemon` (variants: `green_lemon`, `fragrant_lemon`, `seedless_lemon`, `desert_lime`)
- `lime` (variants: `kaffir_lime`, `australian_lime`, `finger_lime`)
- `grapefruit` (variants: `pomelo`, `green_pomelo`, `red_pomelo`)
- `mandarin` (variants: `clementine`, `satsuma_mandarin`, `tangerine`, `ugli_fruit`, `calamondin`, `kumquat`)
- `yuzu`, `citron`, `rangpur`, `trifoliate_orange`

### 4. Berry Fruits (Common, Wild & Niche)
Small pulpy fruits, including true berries and aggregate fruits.
- `blueberry` (variants: `wild_blueberry`)
- `raspberry` (variants: `wild_raspberry`, `andean_raspberry`, `wineberry`, `thimbleberry`, `salmonberry`)
- `blackberry` (variants: `dewberry`, `boysenberry`, `rose_leaf_bramble`)
- `gooseberry` (variants: `cape_gooseberry`, `ceylon_gooseberry`, `indian_gooseberry`)
- `elderberry`, `cloudberry`, `sea_buckthorn`, `lingonberry`, `chokeberry`
- `juniper_berry`, `barberry`, `wolfberry` / `goji_berry`
- `bearberry`, `crowberry`, `buffaloberry`, `nannyberry_fruit`, `snowberry`
- `myrtle_berry`, `quandong`, `kakadu_plum`, `bilberry`
- `currant` (variants: `red_currant`, `black_currant`, `white_currant`, `wild_currant`)
- `acai_berry`, `acerola`, `umbu`, `grumichama`, `pitomba`, `muntingia`

### 5. Culinary Fruits (Botanical Fruits)
These are botanically fruits (containing seeds) but often used as vegetables in cooking. In this dataset, they are strictly classified as **fruits**.
- `tomato` (variants: `beefsteak_tomato`, `cherry_tomato`, `black_tomato`, `orange_tomato`, `tiger_tomato`, `heirloom_tomato`, `wild_tomato`)
- `bell_pepper` (Sweet Pepper Fruit)
- `jalapeno_pepper` (Chili Pepper Fruit)
- `eggplant` (Aubergine Fruit)
- `cucumber` (Gourd Fruit)
- `zucchini` (Summer Squash Fruit)
- `pumpkin` (Winter Squash Fruit)
- `okra` (Lady's Finger Fruit)
- `avocado` (Alligator Pear Fruit)
- `olive` (Olive Fruit)
- `corn` (Maize Kernel/Fruit)
- `vanilla_bean` (Orchid Fruit Pod)
- `cacao_bean` (Cacao Fruit Pod)
- `coffee_cherry` (Coffee Fruit)

### 6. Nut-Fruits & Seed Pods
Hard-shelled fruits containing a seed or kernel. Botanically, these are drupes, capsules, or achenes, classified here as **fruits**.
- `brazil_nut` (Capsule Fruit)
- `pili_nut` (Drupe Fruit)
- `cashew_apple` (with attached Nut Fruit)
- `almond` (Drupe Fruit)
- `walnut` (Drupe Fruit)
- `pecan` (Drupe Fruit)
- `hazelnut` (Nut Fruit)
- `macadamia` (Follicle Fruit)
- `pistachio` (Drupe Fruit)
- `chestnut` (Burr Fruit)
- `acorn` (Cupule Fruit)
- `beechnut` (Mast Fruit)
- `pine_nut` (Cone Seed/Fruit)
- `betel_nut` (Areca Fruit)
- `sunflower_seed` (Achene Fruit)
- `pumpkin_seed` (from Pumpkin Fruit)
- `sesame_seed` (Capsule Fruit)
- `flax_seed` (Capsule Fruit)
- `chia_seed` (Nutlet Fruit)

---

## 🔍 Key Features for Researchers
- **Fine-Grained Recognition**: Distinguishes between subtle variations (e.g., `green_apple` vs `red_apple`, `durian` vs `durian_montong`).
- **Strict Botanical Definition**: Sections 5 and 6 include items like tomatoes, peppers, and walnuts specifically because they are **biological fruits**, ensuring the dataset remains scientifically accurate.
- **Long-Tail Distribution**: Includes many rare classes (e.g., `jaboticaba`, `camu_camu`, `kakadu_plum`) alongside common ones.
- **Global Coverage**: Spans tropical, temperate, and arid region fruits.

> **Usage**: Use the snake_case labels (e.g., `acai_berry`, `durian_montong`, `beefsteak_tomato`, `brazil_nut`) as class names. All 306 classes are treated as **Fruits**.
---
## 🖼️ Sample Visualization

Below are representative examples from the Fruit-306 dataset, showcasing the diversity across **different growth stages**, **environmental conditions**, and **fine-grained categories**.

<div align="center">
  <table>
    <tr>
      <td colspan="3" align="center"><b>🍓 Berray Varieties- Fine-Grained Differences</b></td>
    </tr>
    <tr>
      <td align="center">
        <img src="assets/01.png" width="180" alt="Red Fuji Apple"/>
        <br><i>cloudberry</i>
      </td>
      <td align="center">
        <img src="assets/02.png" width="180" alt="Green Apple"/>
        <br><i>blueberry</i>
      </td>
      <td align="center">
        <img src="assets/03.png" width="180" alt="Snake Apple"/>
        <br><i>dewberry</i>
      </td>
    </tr>
    <tr>
      <td colspan="3" align="center"><b>🍊 Citrus Varieties - Shape & Color Variations</b></td>
    </tr>
    <tr>
      <td align="center">
        <img src="assets/04.png" width="180" alt="Navel Orange"/>
        <br><i>sweet_orange</i>
      </td>
      <td align="center">
        <img src="assets/05.png" width="180" alt="Green Orange"/>
        <br><i>lime</i>
      </td>
      <td align="center">
        <img src="assets/06.png" width="180" alt="Pomelo"/>
        <br><i>green_orange</i>
      </td>
    </tr>
    <tr>
      <td colspan="3" align="center"><b>🌱 Different Growth Stages & Scenes</b></td>
    </tr>
    <tr>
      <td align="center">
        <img src="assets/07.jpeg" width="180" alt="Unripe Strawberry"/>
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
        <img src="assets/10.png" alt="Class Distribution" width="400"/>
        <br><i>Figure: Long-tail distribution across 306 fruit classes</i>
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
---

## 📊 Benchmark Results

### Baseline Performance Comparison

| Model | Top-1 Acc | Top-5 Acc | Params (M) | Latency (ms) |
|:---|:---:|:---:|:---:|:---:|
| GoogleNet [31] | 0.6240 | 0.8164 | 6.8 | 8.2 |
| VGG16 [30] | 0.5860 | 0.7595 | 138.0 | 22.4 |
| Inceptionv3 [32] | 0.1635 | 0.2763 | 23.8 | 14.1 |
| Xception [6] | 0.5830 | 0.7493 | 22.9 | 16.5 |
| InceptionResNetV2 [33] | 0.6734 | 0.8062 | 55.9 | 25.6 |
| **ResNet50 [17]** | **0.6503** | **0.8658** | **25.6** | **12.5** |
| **DenseNet201 [21]** | **0.6850** | **0.8802** | **20.0** | **18.2** |
| **EfficientNetB7 [34]** | **0.6283** | **0.8916** | **66.3** | **45.6** |
| **ViT-B/16 [9]** | **0.5969** | **0.8831** | **86.6** | **32.4** |
| Qwen-VL-Plus [36] | 0.5638 | - | API | 1250 |
| **FruitEnsemble (Ours)** | **0.7049** | **0.9150** | - | **19.8** |

### Ablation Study

| Configuration | Heterogeneous Ensemble | TTA | LLM Arbiter | Top-1 Acc (%) |
|:---|:---:|:---:|:---:|:---:|
| Baseline (ResNet50) | - | - | - | 65.03 |
| + Heterogeneous Ensemble | ✓ | - | - | 68.50 |
| + Test-Time Augmentation | ✓ | ✓ | - | 68.92 |
| **+ LLM Arbitration (Ours)** | ✓ | ✓ | ✓ | **70.49** |

