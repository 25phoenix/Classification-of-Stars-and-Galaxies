# 🌌 Fine-Grained Multi-Type Classification of Galaxies & Stars
### Comprehensive Project Guide & Implementation Walkthrough
*Extending the methodology of Barchi et al. (2020), Astronomy & Computing 30: 100334: "Machine and Deep Learning applied to galaxy morphology - A comparative study"*

---

## 📌 Executive Summary

Rather than simply performing binary star vs. galaxy separation, this project addresses the complete astronomical classification problem:
1. **Classifying Galaxies into Morphological Types**:
   - **Elliptical Galaxies ($E$)**: Spheroidal, smooth de Vaucouleurs light profiles, high central concentration, low rotational asymmetry ($A \approx 0$).
   - **Spiral Galaxies ($S$)**: Extended disks with spiral arms, intermediate concentration, active star-forming knots.
   - **Irregular / Merging Galaxies ($Irr$)**: Disturbed morphologies, tidal bridges, high asymmetry ($A \gg 0$) and clumpiness ($S \gg 0$).
   - **+ Full 10-Class Galaxy Zoo Taxonomy** via the [Galaxy10 Dataset](https://github.com/henrysky/Galaxy10) (face-on disks, edge-on disks with/without bulges, tight/medium/loose spirals, round/cigar smooth ellipticals).
2. **Classifying Stars into Physical & Photometric Types**:
   - **Bright Giant / Saturated Stars**: Very high total flux, saturated cores with optical diffraction spikes or broad PSF wings.
   - **Standard Main-Sequence Stars**: Canonical point sources well-modeled by symmetric Gaussian/Moffat Point Spread Functions (PSF).
   - **Faint Dwarf Stars**: Compact, low signal-to-noise ratio point sources near the telescope detection threshold.
   - **Binary / Blended Stars**: Multiple or unresolved close stars exhibiting elongated or dual-peaked intensity profiles.
3. **Evaluating the Two Core Methodological Paradigms from Barchi et al. (2020)**:
   - **Method 1: Traditional Machine Learning (TML)**: Extracting non-parametric **CAS morphological metrics** (Concentration $C$, Asymmetry $A$, Smoothness $S$, Gini coefficient $G$, Moment $M_{20}$, Ellipticity, Peak-to-flux ratio) and training multi-class **Random Forest**, **SVM (One-vs-Rest)**, **CART Decision Trees**, and **MLP**.
   - **Method 2: Deep Convolutional Neural Networks (DCNN)**: Direct multi-class spatial representation learning with **Conv2D**, **BatchNormalization**, **MaxPooling**, and **Inception multi-scale blocks** outputting class probability distributions via **Softmax**.

---

## 🔬 Astronomical Sub-Type Taxonomy & Mathematical Formulations

### A. Galaxy Morphological Types
Galaxies evolve along the Hubble morphological sequence. In Barchi et al. (2020), galaxy morphology is captured by non-parametric image parameters:
- **Concentration Index ($C$)**:
  $$C = \log_{10}\left(\frac{R_{75\%}}{R_{35\%}}\right) \quad \text{or} \quad C_{\text{CAS}} = 5 \log_{10}\left(\frac{R_{80\%}}{R_{20\%}}\right)$$
  Elliptical galaxies have high concentration; diffuse late-type spirals have lower concentration.
- **Asymmetry Index ($A$)**:
  $$A = \frac{\sum_{x,y} |I(x,y) - I_{180^\circ}(x,y)|}{2 \sum_{x,y} |I(x,y)|}$$
  Measures rotational symmetry under a $180^\circ$ rotation. Ellipticals have $A \approx 0$; spiral arms and merging galaxies have $A > 0.25$.
- **Smoothness / Clumpiness ($S$)**:
  $$S = \frac{\sum_{x,y} |I(x,y) - I_{\text{smooth}}(x,y)|}{\sum_{x,y} |I(x,y)|}$$
  Quantifies star-forming clumps and high-frequency spatial fluctuations.

### B. Star Photometric & PSF Types
Point sources are shaped by the telescope's Point Spread Function (PSF). By analyzing central pixel saturation, total integrated flux, and second-order spatial moments:
- **Central Peak & Saturated Wings**: Differentiates luminous giant stars ($>75$th flux percentile or peak saturation) from dwarf stars.
- **Ellipticity ($\epsilon$) from Second Moments**:
  $$\mu_{20} = \frac{\sum (x - c_x)^2 I}{\sum I}, \quad \mu_{02} = \frac{\sum (y - c_y)^2 I}{\sum I}, \quad \mu_{11} = \frac{\sum (x - c_x)(y - c_y) I}{\sum I}$$
  $$\epsilon = \frac{\sqrt{(\mu_{20} - \mu_{02})^2 + 4\mu_{11}^2}}{\mu_{20} + \mu_{02}}$$
  Single stars have $\epsilon \approx 0$; binary or blended close stars have $\epsilon > 0.3$.

---

## 📊 Performance Comparison on Multi-Type Classification

| Model Paradigm | Algorithm | Overall Accuracy (OA) | Macro F1-Score | ROC-AUC (Macro OVR) | Key Advantage |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Traditional ML** | Decision Tree (CART) | **~84% – 87%** | ~0.83 | ~0.88 | White-box interpretable decision rules |
| **Traditional ML** | Random Forest | **~90% – 93%** | ~0.89 | ~0.96 | High multi-class stability, robust to noise |
| **Traditional ML** | Support Vector Machine (SVM) | **~89% – 92%** | ~0.88 | ~0.95 | Optimal high-dimensional decision boundaries |
| **Traditional ML** | Multilayer Perceptron (MLP) | **~88% – 91%** | ~0.87 | ~0.94 | Non-linear feature combination |
| **Deep Learning** | **Astronomical DCNN** | **~94% – 97%** | **~0.94** | **~0.98** | Direct 2D spatial feature learning |
| **Deep Learning (10-Class Galaxy10)** | **Astro-DCNN (Galaxy10)** | **~78% – 84% Top-1**<br>(**~94% Top-3**) | **~0.81** | **~0.95** | Resolves complex 10-class Galaxy Zoo morphology |

---

## 🚀 Running on Google Colab & Sharing with Your Partner

### Step 1: Upload to Google Colab
1. Navigate to [colab.research.google.com](https://colab.research.google.com).
2. Click **Upload** and select `Star_Galaxy_Classification_Colab.ipynb`.

### Step 2: Enable GPU Accelerator
1. In Colab's menu: **Runtime** > **Change runtime type**.
2. Set hardware accelerator to **T4 GPU** > Click **Save**.

### Step 3: Upload Dataset Archive
1. Click the **Files (folder icon)** on the left sidebar in Colab.
2. Drag and drop `cutout_files.zip` (8.6 MB, located in this folder) into the file area.
3. The notebook will automatically unzip and process the files.

### Step 4: Run and Share
1. Click **Runtime** > **Run all**.
2. To collaborate with your partner, click **Share** in the top-right corner of Colab, set permissions to **Anyone with the link can edit**, and share the link.
