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
# Fruit-306 Dataset: Complete English Variety List

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

## 🧠 Core Highlight: Dynamic MLLM-Guided Arbitration

The most significant methodological innovation of **FruitEnsemble** is our **Dynamic Prompt Generation** strategy. Instead of forcing the MLLM to memorize 306 fruit classes (which often leads to severe hallucinations), our system operates on a **Retrieval-Augmented** mechanism based on our expert-curated botanical database.

When the visual ensemble is uncertain (e.g., confidence < 0.6), it outputs a Top-$K$ ($K=3$) candidate list. Our system dynamically retrieves the botanical descriptions for *only these 3 candidates* and injects them into the prompt. The MLLM then acts purely as an arbitrator using **Chain-of-Thought (CoT)** reasoning.

### 1. Expert-Curated Botanical Database (Snippet)
To support fine-grained multi-modal reasoning, we built a comprehensive textual database containing detailed morphological descriptions for all 306 categories. Below is a snippet of our JSON database demonstrating the level of expert detail—capturing textures, calyx structures, lenticels, and unique shapes:

```json
{
  "001 Ambarella": "Ambarella is an oval or egg-shaped fruit. When ripe, it is golden to orange-yellow. The skin is smooth and slightly waxy, with a distinct persistent calyx at the apex and a slightly depressed navel.",
  "038 Cactus_Fruit": "The cactus fruit is oval or egg-shaped. When ripe, it turns purplish-red to dark red. The skin is densely covered with fine glochids (hair-like spines) and a waxy white bloom, often retaining a dried calyx at the apex, with a slightly uneven surface.",
  "044 Buddha's_Hand": "Buddha's Hand is shaped like open, finger-like segmented lobes. When ripe, it is bright yellow. The skin is rough with distinct bumpy textures and oil glands. The tips of the lobes are often slightly curved, emitting a strong lemon fragrance.",
  "048 Kiwano": "Kiwano is oval or short egg-shaped. When ripe, it is orange-yellow. The skin is covered with hard, cone-shaped horn-like protrusions, resembling a sea urchin. The apex is slightly pointed, with an obvious waxy gloss on the surface.",
  "064 Jaboticaba": "Jaboticaba is spherical. When ripe, it is shiny purplish-black. The skin is smooth with a waxy gloss. The fruits grow densely and directly on the tree trunk. A single fruit is about 1-2.5 cm in diameter, occasionally with subtle light-colored spots on the surface.",
  "080 Mangosteen": "Mangosteen is spherical. When ripe, the rind is dark purplish-red, smooth, and hard. The apex has 4 thick green calyx lobes, and the stem end often has a light brown ring-like mark.",
  "119 Durian": "Durian is ellipsoid-shaped. When ripe, the husk is greenish-yellow to brownish-yellow, densely covered with hard, cone-shaped sharp thorns. The base of the thorns is thick and the tips are sharp. The overall bumpy texture is strong, and the fruit stalk is thick and prominent.",
  "188 Rambutan": "Rambutan is oval to nearly spherical. When ripe, the skin is bright red, densely covered with soft, translucent red fleshy spine-like protrusions (spinterns) about 1-2 cm long, slightly curved at the tips, with bases connecting into a reticulated ridge.",
  "222 Snake_Fruit": "Snake fruit is oval to egg-shaped. When ripe, it is orange-red to brownish-red. The skin is covered with scale-like protrusions resembling snake skin. The texture is rough, and the apex often retains the calyx.",
  "240 Sugar_Apple": "Sugar Apple is conical or heart-shaped. When ripe, it is yellowish-green. The skin is covered with regularly raised knobby scales, resembling the head of Buddha. The tips of the scales are slightly depressed, giving it an overall distinct bumpy feel."
}
```
*(Note: These high-quality textual priors guide the MLLM to focus on subtle discriminative features like textures, lenticels, and calyx structures, rather than generic global patterns).*

### 2. The Dynamic Prompt Template

```python
# MLLM_PROMPT_TEMPLATE
PROMPT_TEMPLATE = """
You are an expert agricultural botanist. An ensemble of visual models has predicted the Top-3 most likely candidate cultivars for the provided fruit image. 

Due to high visual similarity, your expertise is required. Please base your decision **STRICTLY** on the textual botanical priors provided below.

### Retrieved Botanical Priors for Top-3 Candidates:
{dynamic_botanical_priors}

### Your Task (Chain-of-Thought Reasoning):
1. **Visual Attribute Extraction**: Identify discriminative visual features of the fruit in the image (e.g., shape, peel texture, lenticels/oil glands, calyx structure).
2. **Prior Matching & Exclusion**: Compare your extracted features against the botanical priors provided above. Explicitly state which features support a candidate and which contradict a candidate.
3. **Conclusion**: Select the single most accurate cultivar. Do NOT hallucinate classes outside the candidates.

### Output JSON Format:
{{
  "step1_visual_attributes": "...",
  "step2_matching_reasoning": "...",
  "final_prediction": "..."
}}
"""
```

### 3. Full Case Study: Resolving Fine-Grained Ambiguity

*Imagine the visual ensemble is given an image of a Sweet Orange, but due to lighting and camera angle, it is confused between three visually similar citrus fruits.*

**🔹 Step A: Visual Ensemble Output**
The router detects low confidence and outputs Top-3 candidates: `["155 Sweet_Orange", "111 Mandarin", "112 Pomelo"]`.

**🔹 Step B: Dynamic DB Retrieval**
The system automatically queries our JSON botanical database and populates `{dynamic_botanical_priors}` using the exact expert descriptions:

```text
- 155 Sweet_Orange: The sweet orange is perfectly spherical or slightly oblate. When ripe, the peel is orange-red to orange-yellow. The surface is smooth and delicate, with fine oil glands, occasionally showing shallow grooves and a small depression at the navel.
- 111 Mandarin: The mandarin is oblate or nearly spherical. When ripe, the peel is orange-red to bright orange. The surface is smooth with slightly raised oil glands, often with fine reticulated patterns. The apex is slightly depressed, and the stem base slightly protrudes.
- 112 Pomelo: The pomelo is oblate or pear-shaped. When ripe, the peel is yellowish-green to bright yellow. The surface is rough and highly bumpy, densely covered with obvious oil glands, often with a waxy gloss. The fruit apex has radial shallow grooves.
```

**🔹 Step C: MLLM Arbitration Output (Actual System Response)**

```json
{
  "step1_visual_attributes": "The fruit in the image has a perfectly spherical shape with an orange-yellow skin. The surface texture is very smooth and delicate, with very fine, subtle oil glands. The stem base does not protrude, and the skin does not appear rough or pear-shaped.",
  "step2_matching_reasoning": "The 'smooth and delicate surface with fine oil glands' and 'spherical shape' directly support the prior for '155 Sweet_Orange'. The lack of a 'protruding stem base' and 'flat round shape' contradicts the description of '111 Mandarin'. Furthermore, the absence of a 'pear shape' and 'rough, highly bumpy surface with obvious oil glands' clearly excludes '112 Pomelo'.",
  "final_prediction": "155 Sweet_Orange"
}
```

> *Result: The MLLM successfully corrects the ensemble's uncertainty by grounding its visual reasoning strictly in the retrieved expert botanical text!*
