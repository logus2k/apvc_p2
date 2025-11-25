# Data Preparation Phase - Planning Notes

---

# **Phase 1** - DU to DP Decisions

## **Step 1.1** - DU Summary Tables

From DF_VIEW:

* global contrast distribution
* entropy distribution
* kurtosis tail
* local variance distribution
* high-frequency distribution

Goal: **identify which DU metrics indicate meaningful inconsistencies**.

**Deliverable**

Subsection: *“DU Summary and Implications for Data Preparation”*

## **Step 1.2** - Preprocessing Goals From DU

Based on DU results:

1. **Contrast varies?**
   mild normalization or CLAHE (if needed)
2. **Sharpness/entropy outliers?**
   consider denoising or exclude worst cases
3. **Local variance outliers?**
   identify noise vs smoothing problems
4. **High-frequency tail?**
   avoid overaggressive contrast enhancement
5. **Dimension variations?**
   set aspect-ratio handling policy (crop vs pad vs hybrid)

**Deliverable**

Subsection: *“Data Preparation Requirements Inferred from DU”*

---

# **Phase 2** - cxray Integration

## **Step 2.1** - Load DU Metrics Into cxray (we start shortly)

Step 1 - Export JSON
Step 2 - Load DU metrics on cxray backend

Enable cxray to:

* filter images by DU criteria
* visually inspect outliers
* compare “before vs after” for preprocessing plans

## **Step 2.2** - Use cxray to validate preprocessing decisions

Run cxray with:

* cropping candidates
* padding policies
* intensity transforms (e.g., soft CLAHE)
* zoom levels
* brightness/contrast changes

**Goal**

Make preprocessing decisions that:

* align with DU statistical distributions,
* preserve diagnostic information,
* stabilize gradient flow for training.

**Deliverable**

Screenshots and examples shown in the notebook (optional)

## **Step 2.3 - Finalize Preprocessing Preset**

cxray can produces *preprocessing presets*, e.g.:

```json
{
  "crop_top_pct": 0.05,
  "crop_bottom_pct": 0.08,
  "crop_left_pct": 0.02,
  "crop_right_pct": 0.02,
  "apply_clahe": true,
  "clahe_clip_limit": 2.0,
  "resize_target": [224, 224],
  "padding_strategy": "center-pad"
}
```

One or several cxray presets are used to create DF_TRAIN.

---

# **Phase 3** - DF_TRAIN

Apply the validated preprocessing to the training dataset.

## **Step 3.1** - Preprocessing Pipeline

* Apply cropping
* Apply padding or aspect-ratio policy
* Apply contrast normalization or CLAHE
* Apply resizing
* Apply denoising if needed
* Save preprocessed images (optional) or apply in tf.data pipeline

Record transformations in the DF_TRAIN rows:

```
processed_image_path
preprocessing_applied: {...}
```

## **Step 3.2** - Recompute DU metrics on DF_TRAIN (optional)

Purpose:

* ensure preprocessing reduced variance
* ensure metrics now fall within acceptable uniform ranges
* detect whether preprocessing introduced new distortions

This can validate that DP improved dataset consistency for Modeling.

---

# **Phase 4** - Data Augmentation Strategy

Data Augmentation can be designed based on DU insights.

## **Step 4.1** - Define augmentation limits

Based on DU:

* rotation: small degrees only
* zoom: avoid changing local variance too much
* brightness: within entropy-stable bounds
* flips: horizontal only (no vertical for CXRs)
* random crops: constrained by cxray decisions

## **Step 4.2** - Implement augmentation pipeline

Connect directly to:

* TF/Keras `ImageDataGenerator`
* or `tf.data` augmentation block

## **Step 4.3** - Validate augmentations visually

Use cxray or notebook plotting to ensure:

* augmentations do not violate radiological plausibility
* anatomical structures remain intact

---

# **Phase 5** - DP transition to Modeling

## **Step 5.1** - Define tf.data pipelines per model

* apply preprocessing
* convert to model-specific normalization
* batch, shuffle, prefetch

## **Step 5.2** - Build Model Baselines

Using:

* VGG16
* ResNet50
* EfficientNet

## **Step 5.3** - Training and Evaluation

TBD.

---
