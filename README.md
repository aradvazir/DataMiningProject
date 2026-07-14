# Data Mining — Course Projects

Four self-contained **data mining / machine learning** projects, each taking a different dataset from raw CSV to a tuned, cross-validated, error-analysed classifier. Together they cover the full classical ML toolbox: **PCA (hand-derived and via scikit-learn), autoencoders, K-Means, SVM, KNN, Naive Bayes, Random Forests, AdaBoost, MLPs, semi-supervised pseudo-labelling, and stacking.**

Every project follows the same disciplined loop, which is really the point of the repo:

> **explore → preprocess → reduce dimensionality → grid-search hyperparameters with 5-fold CV → fit → evaluate on *both* test and train with a confusion matrix and per-class precision/recall/F1.**

Reporting train-set metrics next to test-set metrics in every single project is a deliberate habit here: it makes overfitting visible instead of hypothetical (see the Random Forest scoring 100% on train vs. 95.6% on test, or the MLP at 98.8% vs. 87.5%).

---

## Table of Contents

- [Repository structure](#repository-structure)
- [Setup](#setup)
- [Datasets you need to supply](#datasets-you-need-to-supply)
- [Project 1 — Forest Cover Type with Random Forests](#project-1--forest-cover-type-with-random-forests-decisiontree-randomforestipynb)
- [Project 2 — Fashion-MNIST: PCA from Scratch + Four Classifiers](#project-2--fashion-mnist-pca-from-scratch--four-classifiers-fashion-mnist)
- [Project 3 — Semi-Supervised MNIST: Autoencoder + K-Means Pseudo-Labelling](#project-3--semi-supervised-mnist-autoencoder--k-means-pseudo-labelling-mnist-handwritten)
- [Project 4 — Network Intrusion Detection on UNSW-NB15](#project-4--network-intrusion-detection-on-unsw-nb15-network-intrusion-detection)
- [Results at a glance](#results-at-a-glance)
- [Known issues, quirks and caveats](#known-issues-quirks-and-caveats)
- [Ideas for improvement](#ideas-for-improvement)
- [Author](#author)

---

## Repository structure

```
DataMiningProject/
│
├── DecisionTree-RandomForest.ipynb          # Forest Cover Type (covtype) — Random Forest grid search
│
├── Fashion-MNIST/
│   ├── Fashion-MNIST.ipynb                  # PCA from scratch + SVM / GaussianNB / KNN / MLP
│   └── Vazirpanah_Arad-610399182-HW2.pdf    # Written report
│
├── MNIST-HandWritten/
│   ├── MNIST-handwritten.ipynb              # PyTorch autoencoder + K-Means pseudo-labelling (semi-supervised)
│   └── Vazirpanah-Arad-6103991982-HW3.pdf   # Written report
│
└── Network-Intrusion-Detection/
    ├── Network-Intrusion-Detection.ipynb    # UNSW-NB15 multi-class IDS + stacking ensemble
    └── Report.pdf                           # Full 25-page report, incl. comparison with a published paper
```

Note: the first notebook is written in **Persian**; the other three are in English.

---

## Setup

Python 3.8+ with Jupyter. A minimal environment:

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate

pip install jupyter numpy pandas matplotlib seaborn scikit-learn
pip install torch                                     # only for MNIST-HandWritten
```

Several cells are genuinely slow — the `covtype` Random Forest grid search (12 configs × 5 folds on ~400k rows) and the Fashion-MNIST linear SVM in particular. The linear-kernel SVM was **abandoned mid-run** and is left commented out in the notebook for exactly that reason.

---

## Datasets you need to supply

No data is committed. Drop the following next to each notebook before running it:

| Notebook | Expects | Dataset |
|---|---|---|
| `DecisionTree-RandomForest.ipynb` | `covtype.csv` | **Forest Cover Type** — 581,012 rows × 54 features, 7 cover-type classes (UCI / Kaggle) |
| `Fashion-MNIST.ipynb` | `fashion-mnist_train.csv`, `fashion-mnist_test.csv` | **Fashion-MNIST** — 60k train / 10k test, 28×28 greyscale, 10 clothing classes |
| `MNIST-handwritten.ipynb` | `labeled_train_set.csv` (18k), `unlabeled_train_set.csv` (42k), `test_set.csv` (10k) | **MNIST** digits, pre-split into a labelled / unlabelled / test partition for the semi-supervised exercise |
| `Network-Intrusion-Detection.ipynb` | `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv` | **UNSW-NB15** — 175,341 train / 82,332 test flows, 44 columns, 10 attack categories |

All CSVs are expected in the notebook's working directory.

---

## Project 1 — Forest Cover Type with Random Forests (`DecisionTree-RandomForest.ipynb`)

Predict which of 7 forest cover types occupies a 30×30 m patch, from 54 cartographic features (elevation, slope, hillshade, soil type, wilderness area…).

**Pipeline**

1. Split off the label column; **normalize only the first 10 columns** — those are the continuous cartographic measurements. Columns 11–54 are already one-hot binary indicators (wilderness area + soil type), so scaling them would be meaningless.
2. **Stratified** 70/30 train/test split — important here, because the class distribution is heavily skewed (class 2 ≈ 198k rows vs. class 4 ≈ 1.9k in the training split).
3. **Grid search over 12 configurations** with 5-fold cross-validation:

   | Hyperparameter | Values |
   |---|---|
   | `n_estimators` | 80, 100, 120 |
   | `max_features` | 5, 7 |
   | `criterion` | `gini`, `entropy` |

**Result.** All 12 configurations land within a **0.3-point band** (CV accuracy 0.9469 – 0.9497) — a useful negative result in itself: on this dataset a Random Forest is remarkably insensitive to these knobs. The best is `n_estimators=120, max_features=7, criterion="entropy"` (CV = 0.9497).

| | Test | Train |
|---|---:|---:|
| **Accuracy** | **0.9557** | **1.0000** |

The perfect training accuracy is exactly what you'd expect from a forest of unpruned trees, and the notebook says as much — the gap to 95.6% on held-out data is the model's real generalization error, not a bug.

**Per-class test metrics** (classes 1–7):

| Class | Precision | Recall | F1 | Support |
|---:|---:|---:|---:|---:|
| 1 | 0.964 | 0.947 | 0.955 | 63,552 |
| 2 | 0.953 | 0.972 | 0.962 | 84,991 |
| 3 | 0.941 | 0.962 | 0.951 | 10,726 |
| 4 | 0.914 | 0.830 | 0.870 | 824 |
| 5 | 0.940 | **0.803** | 0.866 | 2,848 |
| 6 | 0.930 | 0.890 | 0.910 | 5,210 |
| 7 | 0.974 | 0.952 | 0.963 | 6,153 |

The weak spot is **recall on class 5** (0.80) — a fifth of its instances are scattered into other classes, mostly class 2, which is unsurprising given class 2 has ~30× more training examples. Precision stays high everywhere, so the model is *cautious* rather than *wrong* about the minority classes.

---

## Project 2 — Fashion-MNIST: PCA from Scratch + Four Classifiers (`Fashion-MNIST/`)

Classify 28×28 greyscale clothing images (T-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot) into 10 classes.

### PCA, twice

The notebook first derives PCA **by hand** to show it isn't magic:

1. Mean-center each of the 784 pixel columns.
2. Compute the covariance matrix directly: `X_cov = Xᵀ X / (n − 1)`.
3. Eigendecompose it with `np.linalg.eig` — eigenvectors are the principal components, eigenvalues their variances.
4. Project onto the top 2 PCs and scatter-plot, coloured by class.

It then repeats the projection with `sklearn.decomposition.PCA(n_components=2)` and confirms the plots match — a nice verification step that most coursework skips.

For the actual modelling, PCA is set to retain **96% of the variance**, which compresses **784 → 226 dimensions** (a ~3.5× reduction with almost no information loss) and makes the SVM and KNN tractable.

### Classifier bake-off (all tuned by 5-fold CV on the PCA features)

| Model | Tuning searched | Best CV | **Test acc** | Train acc |
|---|---|---:|---:|---:|
| **SVM** | kernel: `poly`, `rbf` (linear abandoned — too slow) | 0.892 (`rbf`) | **0.8976** | 0.9168 |
| **MLP** | layers `[75, 100, (75,150), (100,150), (200,200)]`, then learning-rate schedule, then activation, then epochs | 0.869 | **0.8746** | 0.9878 |
| **KNN** | `k ∈ {3, 5, 7}` | 0.861 (`k=5`) | **0.8669** | 0.9045 |
| **Gaussian NB** | — | 0.733 | **0.7359** | 0.7354 |

**Reading the results.**

- The **RBF SVM wins on test data** (89.8%) with the smallest train–test gap of the top three (only ~2 points), making it the best-generalizing model here.
- The **MLP has by far the largest gap** — 98.8% train vs. 87.5% test, an 11-point spread that is textbook overfitting. It memorizes the training set without translating that into held-out accuracy.
- **Gaussian Naive Bayes collapses to ~73%**, and its train and test accuracies are *identical*. That's the signature of a model that is underfitting, not overfitting: its independence assumption (and Gaussian likelihood over PCA components) simply cannot represent the data, so it fails equally everywhere.
- The MLP hyperparameter search is done **greedily** — layer sizes first, then learning-rate schedule, then activation, then epoch count — each stage fixing the previous winner. Cheap, but it can miss interactions that a full grid would catch.

---

## Project 3 — Semi-Supervised MNIST: Autoencoder + K-Means Pseudo-Labelling (`MNIST-HandWritten/`)

The most conceptually interesting project. **Only 18,000 of the 60,000 training digits carry labels.** The goal is to exploit the other 42,000 unlabelled images anyway.

### Step 1 — Representation learning with an autoencoder (PyTorch)

A symmetric fully-connected autoencoder squeezes each image through a **36-dimensional bottleneck**:

```
784 → 128 → 64 → 36  (encoder)
36 → 64 → 128 → 784  (decoder, Sigmoid output)
```

Trained with **MSE loss** and **SGD (lr=0.01, momentum=0.9)** for 25 epochs. This is unsupervised — it never sees a label — so the learned 36-d code is a general-purpose compression of digit shape, and the reconstruction loss curve is plotted to confirm convergence. The trained weights are checkpointed to `Autoencoder.pth` so the (slow) training doesn't have to be repeated.

### Step 2 — Supervised baseline on the encoded features

An `MLPClassifier` is grid-searched over 12 combinations (layers × learning-rate schedule × activation) on the 36-d codes, then a `(100, 75, 50)` / `invscaling` / `tanh` network is trained.

| | Test | Train |
|---|---:|---:|
| **Accuracy** | **0.9483** | **1.0000** |

**94.8% test accuracy from a 36-dimensional code** — the autoencoder threw away 95% of the input dimensions and kept nearly all the class-relevant signal.

### Step 3 — Pseudo-labelling the unlabelled 42,000

This is the clever bit:

1. Run **K-Means with 500 clusters** on the raw unlabelled images. 500 ≫ 10 on purpose — over-clustering means each cluster is likely to be *pure* (one digit written one way) even though many clusters share a digit.
2. For each cluster, sample **10 random members**, encode them, and ask the trained MLP to classify them.
3. Take a **majority vote** and stamp that label on the entire cluster.
4. Reconstruct and visualize the mean image of each induced class as a 2×5 grid of `28×28` digits — a visual sanity check that the mapping actually recovered the ten digits.

The induced label counts come out remarkably balanced (3,740 – 4,805 per digit, summing exactly to 42,000), which is strong evidence the cluster→digit mapping largely worked.

### Step 4 — Retrain on all 60,000

The MLP is retrained on the labelled 18k **plus** the pseudo-labelled 42k. Cross-validated accuracy on the combined set is **0.9259**.

**An honest caveat**, which the notebook doesn't flag: the final "test" evaluation (0.9792) is computed on `test_set ∪ unlabeled_train_set`, scoring the model against **its own pseudo-labels** on 42k of those 52k rows. That number is circular and should not be read as generalization. The trustworthy figures are the 0.9483 clean test accuracy from Step 2, and the 0.9259 CV score from Step 4.

---

## Project 4 — Network Intrusion Detection on UNSW-NB15 (`Network-Intrusion-Detection/`)

Multi-class classification of network flows into **10 categories** — `Normal`, `Generic`, `Exploits`, `Fuzzers`, `DoS`, `Reconnaissance`, `Analysis`, `Backdoor`, `Shellcode`, `Worms` — from 44 flow features (duration, packet counts, byte counts, TTLs, jitter, TCP timings, connection-tracking counts, …). This is the most thorough project in the repo, and it comes with a **25-page report** benchmarked against a published paper.

### Exploration

40 feature-vs-class scatter plots are generated (in three batches, grouped by attack category) plus class-frequency bar charts for train and test. The conclusion drawn is refreshingly blunt: **most features show no separable pattern at all**. A handful (`trans_depth`, `ct_srv_src`, `ct_dst_ltm`, `ct_dst_src_ltm`, `ct_srv_dst`) partially separate some classes. The report explicitly predicts, before training anything, that accuracy will be underwhelming — and it is.

### Preprocessing

1. Encode the 10 attack categories to integers via `np.unique` index.
2. **Pull out** the three categorical columns (`proto`, `service`, `state`) and drop `id`.
3. **MinMax-scale** the remaining numeric columns — with the scaler `fit` on train only and merely `transform`ed on test. (Correct: no test statistics leak into training.)
4. Re-attach the categoricals and **one-hot encode** with `pd.get_dummies`, then `reindex` the test frame to the training columns with `fill_value=0` — this handles protocol/service values that appear in only one split, a step that's easy to get wrong.
5. **PCA retaining 98% of variance.**

### Two roads not taken (deliberately)

The reference paper improved its numbers by (a) **dropping the four rare classes** and (b) **oversampling `Normal`** to match the test distribution. The report refuses to do (b), and the reasoning is worth quoting in spirit: that decision *is made by looking at the test set*, so the resulting gain isn't real — it wouldn't survive contact with live traffic. The oversampling line is left in the notebook, commented out. This is an unusually mature call for coursework, and it's the main reason the numbers below look worse than the paper's.

### Model comparison

Each model was hyperparameter-tuned by 5-fold CV, then fit and scored on **both** splits:

| Model | Configuration | CV | **Test acc** | Train acc |
|---|---|---:|---:|---:|
| SVM | `poly` (rbf 0.757 > sigmoid 0.662 in search) | 0.757 | 0.6996 | 0.7767 |
| KNN | `k = 7` (best of 3/5/7) | 0.732 | 0.7152 | 0.8081 |
| Random Forest | 75 trees, `max_features=7`, entropy | 0.754 | 0.7274 | **0.9075** |
| MLP | `(60, 45, 35, 27)`, adaptive, relu | 0.778 | 0.7366 | 0.8280 |
| AdaBoost | defaults | — | 0.6014 | 0.7042 |
| **MLP′** | `(60, 45, 35, 27, 20, 15)`, adaptive, relu | 0.778 | **0.7492** | 0.8285 |

### Stacking ensemble

A hand-rolled stacking cascade: PCA again → **KNN** predicts and its prediction is **appended as a new feature column** → **Random Forest** trains on the augmented matrix and appends *its* prediction → a final **MLP** (or KNN) trains on top.

This produces the repo's most instructive failure. Cross-validated accuracy leaps to **0.90**, and train accuracy to 0.907 — but **test accuracy stays flat at 0.728**. The reason is that the stacked feature columns are the base models' predictions *on the very rows they were fit to*, so those columns are near-perfect copies of the label during training and useless at inference time. The report notices the symptom ("it might be over fitted…") without naming the mechanism. **Proper stacking requires out-of-fold predictions** (`cross_val_predict`, or scikit-learn's `StackingClassifier`) — otherwise you are simply leaking the target into the feature matrix.

### Final result

The best honest classifier is a deeper **MLP at 76.6% test accuracy** (82.1% train). The report closes by comparing its 10×10 confusion matrix against the paper's 6×6 (the paper having dropped the rare classes) and its 84.24% accuracy, and correctly attributes most of the gap to the paper's test-set-informed preprocessing rather than to a better model.

---

## Results at a glance

| Project | Dataset | Best model | Test accuracy |
|---|---|---|---:|
| Forest Cover Type | `covtype` (581k × 54, 7 classes) | Random Forest (120 trees, entropy) | **95.6%** |
| Fashion-MNIST | 60k/10k, 784 px → 226 PCs | RBF SVM | **89.8%** |
| Semi-supervised MNIST | 18k labelled + 42k unlabelled | Autoencoder(36-d) + MLP | **94.8%** |
| Network Intrusion | UNSW-NB15, 10 classes | MLP `(60,45,35,27,20,15)` | **76.6%** |

The spread is the story. Cover-type and MNIST are *easy* — the features carry the signal, and almost any tuned model gets there. UNSW-NB15 is *hard*, and no amount of model-shopping fixes it, because (as the exploration plots showed up front) the features barely separate the classes. Recognizing which situation you're in before you start tuning is most of the skill.

---

## Known issues, quirks and caveats

Worth knowing before running or reusing this code:

- **`DecisionTree-RandomForest.ipynb` contains no decision tree.** Despite the filename, only `RandomForestClassifier` is used. There's also a stray cell fitting a throwaway 20-tree forest between the grid search and the final model.
- **Its final plot has the axes swapped** — `plt.plot(scores, [1..12])` puts accuracy on x and the config index on y.
- **MNIST autoencoder trains one sample at a time** (`for i in range(len(X_ltrain))`, no `DataLoader`, no batching) — 25 epochs × 18,000 samples = 450,000 optimizer steps of batch size 1. Batching would cut this from hours to minutes.
- **MNIST Step-4 "test accuracy" (0.9792) is not a real test score** — it evaluates on `test ∪ unlabeled`, where 42k of the 52k labels are the model's own pseudo-labels. See the caveat in Project 3.
- **MNIST cell 46 predicts with the wrong model** — it fits `mlp_model` but then calls `mlp.predict(...)` (the *old* classifier), so the 0.944 it prints doesn't measure what the surrounding markdown claims.
- **UNSW-NB15 AdaBoost is cross-validated on the wrong estimator** — `cross_val_score(mlp, ...)` instead of `cross_val_score(adaboost, ...)`, so the printed 0.779 is the MLP's score, not AdaBoost's. AdaBoost's real numbers are the 0.60 test / 0.70 train accuracies.
- **The stacking ensemble leaks the target** (in-sample predictions used as features). Its inflated 0.90 CV score is not real. Use `cross_val_predict` / `StackingClassifier`.
- **The final UNSW-NB15 comparison plots** list eight algorithm names (`algorithms`) against an accuracy array that only receives six appends in a clean top-to-bottom run — the plot only renders if cells were executed out of order.
- **Fashion-MNIST: the written report ranks MLP first**, but the recorded numbers put the RBF SVM ahead on test data (89.8% vs. 87.5%). The MLP is only ahead on *training* accuracy, which is precisely the metric not to rank on.
- **Fashion-MNIST re-centers train and test with their own means** (`X_test - mean_X_test`) rather than applying the training mean to both. Harmless on a large, balanced test set, but it's the same class of leak that the UNSW-NB15 notebook is careful to avoid.
- Random seeds are not fixed anywhere (`train_test_split`, `KMeans`, `MLPClassifier`), so re-runs will shift the numbers by a few tenths of a point.

---

## Ideas for improvement

- Replace the greedy, sequential hyperparameter searches with `GridSearchCV` / `RandomizedSearchCV` and a `Pipeline`, so scaling and PCA are fit **inside** each CV fold rather than on the full training set.
- Batch the autoencoder training with a `DataLoader` (batch size 128, Adam) — same result, orders of magnitude faster — and try a **convolutional** autoencoder, which should beat 36 dense dimensions on image data comfortably.
- Rebuild the UNSW-NB15 stack with `StackingClassifier` (out-of-fold predictions) and see whether the gain survives. It probably won't — and demonstrating that cleanly would be a stronger result than the current inflated CV number.
- On UNSW-NB15, address the class imbalance *from the training set only* — class weights, or SMOTE applied inside the CV folds — and report **macro-F1** instead of accuracy, which is the metric that actually matters for intrusion detection (catching Worms matters more than nailing another `Normal` flow).
- Fix random seeds throughout and pin dependencies in a `requirements.txt`.
- Skipping `covtype`'s Random Forest through a `HistGradientBoostingClassifier` or LightGBM would likely push past 96–97% at a fraction of the training time.

---

## Author

Coursework by **Arad Vazir Panah** (student no. 610399182).

Written reports accompany three of the four projects and go well beyond the code — the [UNSW-NB15 report](Network-Intrusion-Detection/Report.pdf) in particular includes the full exploratory analysis, every confusion matrix, and a section-by-section comparison against a published paper on the same dataset.

---

## License

No license file is currently included, which means the code is under exclusive copyright by default. If you'd like others to reuse it, consider adding a `LICENSE` (MIT is a common choice for coursework).
