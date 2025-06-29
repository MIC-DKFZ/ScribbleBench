# ScribbleBench

**ScribbleBench** is a comprehensive benchmark for evaluating the generalization capabilities of 3D scribble-supervised medical image segmentation methods. It spans seven diverse datasets across multiple anatomies and modalities and provides realistic, automatically generated scribble annotations.

This repository provides:
- A guide on how to setup the ScribbleBench benchmark using the original dataset sources and our ScribbleBench scribbles.
- Our scribble generation code to create realistic interior and boundary scribbles heuristics.
- An evaluation script to evaluate your method using ScribbleBench.
- A reference to our scribble baseline nnnUNet+pL
- A scribble annotation protocol for domain experts that can be used as guidance to quickly annotate new datasets manually.

ScribbleBench was introduced in our MICCAI 2025 paper:  
**“Revisiting 3D Medical Scribble Supervision: Benchmarking Beyond Cardiac Segmentation”**  
Authors: Karol Gotkowski, Klaus H. Maier-Hein, Fabian Isensee


## 📦 Benchmark Setup

ScribbleBench includes scribbles for the following 7 public datasets:
- ACDC
- MSCMR
- WORD
- AMOS2022 (Task2)
- KiTS2023
- LiTS
- BraTS2020

### 📥 Download Datasets

We are currently preparing a script to setup the ScribbleBench benchmark which automatically downloads the datasets from their original sources alongside our scribbles and preprocesses all data. The script should be finished over the next couple days. If you want to get started with ScribbleBench now, you can download our ScribbleBench scribbles [here](https://syncandshare.desy.de/index.php/s/YT6Ky4CyBNKrmPw) and setup the datasets yourself.


## 🛠️ Scribble Generation

You can use our script to generate scribbles for your own 3D medical segmentation datasets. The script supports:
- Interior scribbles using NURBS curves.
- Boundary scribbles based on partial contours.
- Foreground/background slice balancing.
- Multiprocessing for efficient processing of large datasets.

### 🚀 Run Scribble Generation

```bash
python generate_scribbles.py \
  --input path/to/dense_segmentations \
  --output path/to/save_scribbles \
  --num_labels 4 \
  --conf scribble_conf.yml \
  --processes 8
```

**Optional arguments:**

* `--name` → specify one or more file names to process (omit `.nii.gz`)
* `--disable_ignore` → disables marking unlabeled voxels with an ignore label

## 📊 Evaluation

You can evaluate your segmentation predictions using the provided script:

```bash
python evaluation.py \
  --gt_dir path/to/ground_truth \
  --pred_dir path/to/predictions \
  --num_labels 4 \
  --processes 8
```

## 🧠 Scribble Baseline nnUNet+pL

Our scribble baseline nnUNet+pL is implemented in the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) framework itself. It is there referred to as "ignore label" and is described [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/ignore_label.md).

## 📋 Scribble Annotation Protocol

You can also manually create your own scribbles for new datasets by following this lightweight annotation protocol. These human-created scribbles can be used directly to train a model using the same methods as with automatically generated ones.

### ✏️ Instructions

Given a 3D image **I** in your dataset:
- For each axial slice **S** in **I**:
  - For each class **C** present in slice **S**:
    - Select a single **connected component (CC)** of class **C** in **S**
    - For that component **CC**, draw:
      - One **interior scribble**
      - One **boundary scribble**

Note: Do not ignore the background class! Also include a good number of pure background slices.

#### 🟢 Interior Scribble
- Must be drawn **inside the component CC**.
- Should be placed roughly **in and around the center area** of the component.
- Ideal length is **comparable to the diameter or extent** of the component.
- Can be any arbitrary shape (straight, curved, etc.) as long as it lies **fully within the component**.

#### 🔵 Boundary Scribble
- Should trace **a portion (15%–100%)** of the **inner boundary** of the component CC.
- Should ideally follow the actual boundary as closely as possible.
- A **1–3 voxel inward offset** is acceptable, but **closer to the true boundary is better**.
- This scribble helps the model capture **boundary details** during learning.

Following this protocol allows quick and efficient labeling of 3D datasets using just a few sparse lines per class and slice, while maintaining strong training performance.



## 📄 Citation

If you use ScribbleBench or our scribble generation code, please cite:

```bibtex
@inproceedings{gotkowski2025scribblebench,
  title     = {Revisiting 3D Medical Scribble Supervision: Benchmarking Beyond Cardiac Segmentation},
  author    = {Karol Gotkowski and Klaus H. Maier-Hein and Fabian Isensee},
  booktitle = {International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year      = {2025}
}
```


## 📬 Contact

For questions, suggestions, or contributions, feel free to open an issue or contact [karol.gotkowski@dkfz.de](mailto:karol.gotkowski@dkfz.de).
