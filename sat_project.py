# %%
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import joblib
import warnings
warnings.filterwarnings('ignore')

# %%
# this function shows the satellite rgb image next to its land cover mask
def show_rgb_and_mask(rgb_path, mask_path):
    
    with rasterio.open(rgb_path) as f:
        r = f.read(1).astype(float)
        g = f.read(2).astype(float)
        b = f.read(3).astype(float)

    # stretch each band so image looks natural
    def stretch(band):
        low  = np.percentile(band[band > 0], 1)
        high = np.percentile(band[band > 0], 99)
        band = np.clip(band, low, high)
        band = (band - low) / (high - low)
        return band

    rgb = np.stack([stretch(r), stretch(g), stretch(b)], axis=-1)
    # gamma correction and white balance to look closer to real colors
    rgb = np.power(rgb, 0.75)
    rgb[:, :, 0] *= 0.85
    rgb[:, :, 1] *= 0.90
    rgb[:, :, 2] *= 0.85
    rgb = np.clip(rgb, 0, 1)

    with rasterio.open(mask_path) as f:
        mask = f.read(1)

    colors = ['black', 'darkgreen', 'yellow', 'darkblue', 'gray']
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm([0, 1, 2, 3, 4, 5], cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(rgb)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')

    im = axes[1].imshow(mask, cmap=cmap, norm=norm)
    cbar = plt.colorbar(im, ax=axes[1], ticks=[0.5, 1.5, 2.5, 3.5, 4.5])
    cbar.ax.set_yticklabels(['Unknown', 'Greenery', 'Sand', 'Water', 'Cement'])
    axes[1].set_title('Land Cover Mask')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

# BASE_SAMPLE = r'S2_ML_Dataset_V2\WadiElGemal'
# show_rgb_and_mask(BASE_SAMPLE + r'_RGB.tif', BASE_SAMPLE + r'_mask_ESRI.tif')

# %%
DATA_FOLDER = r'S2_ML_Dataset_V2'
MODEL_FOLDER = r'models'
os.makedirs(MODEL_FOLDER, exist_ok=True)

CLASS_NAMES = ['Greenery', 'Sand', 'Water', 'Cement']
CLASS_LABELS = [1, 2, 3, 4]


# %%

TRAIN_LOCS = [
    # sand / desert locations
    'BaharyaOasis', 'EasternDesert', 'FarafraOasis', 'KhargaOasis',
    'LibyanBorder', 'SinaiDesert', 'WhiteDesert',
    # greenery / farming locations
    'AssiutFarms', 'BeheiraNileDelta', 'DakhlaOasis', 'DamietaFarms',
    'HawaraFayoum', 'MiniaFarms', 'NileDelta', 'SohagFarms',
    'WadiNatrun', 'ZagazigFarms',
    # cement / urban locations
    'AlexandriaPort', 'CairoDowntown', 'CairoUniv', 'FayoumFarms',
    'GizaPlateau', 'HelwanIndustrial', 'MansouraCity', 'NaserCity',
    # water locations
    'AbuSimbelDesert', 'BardawilLake', 'GreatBitterLake', 'GulfSuez',
    'ManzalaLake', 'NasserLake', 'RedSeaHurghada', 'SaddElAly', 'SuezCanal',
]

VAL_LOCS = [
    'IconicTower', 'QattaraDepression',
    'KarnakLuxor', 'SiwaOasis',
    'PortSaidCity', 'ZagazigCity',
    'GulfAqaba', 'MariutLake',
]

TEST_LOCS = [
    'QenaFarms', 'TantaCity',
    'NileCairo', 'SuezCity',
    'WadiElGemal', 'WadiRum',
    'QarunLake', 'TimsahLake',
]

# %%
# # Categorize ALL locations by dominant land cover type
# DESERT_LOCS = [
#     'BaharyaOasis', 'EasternDesert', 'FarafraOasis', 'KhargaOasis',
#     'LibyanBorder', 'SinaiDesert', 'WhiteDesert', 'QattaraDepression',  # was val
#     'SiwaOasis',  # was val
#     'WadiRum',    # was test
# ]

# GREENERY_LOCS = [
#     'AssiutFarms', 'BeheiraNileDelta', 'DakhlaOasis', 'DamietaFarms',
#     'HawaraFayoum', 'MiniaFarms', 'NileDelta', 'SohagFarms',
#     'WadiNatrun', 'ZagazigFarms', 'KarnakLuxor',  # was val
#     'QenaFarms',  # was test
# ]

# URBAN_LOCS = [
#     'AlexandriaPort', 'CairoDowntown', 'CairoUniv', 'FayoumFarms',
#     'GizaPlateau', 'HelwanIndustrial', 'MansouraCity', 'NaserCity',
#     'IconicTower',   # was val
#     'TantaCity',     # was test
#     'SuezCity',      # was test
#     'PortSaidCity',  # was val
#     'ZagazigCity',   # was val
# ]

# WATER_LOCS = [
#     'AbuSimbelDesert', 'BardawilLake', 'GreatBitterLake', 'GulfSuez',
#     'ManzalaLake', 'NasserLake', 'RedSeaHurghada', 'SaddElAly', 'SuezCanal',
#     'GulfAqaba',   # was val
#     'MariutLake',  # was val
#     'NileCairo',   # was test — urban river
#     'QarunLake',   # was test
#     'TimsahLake',  # was test
# ]

# # Now split each category ~70/15/15
# import random
# random.seed(42)

# def split_category(locs, val_n=2, test_n=2):
#     shuffled = locs.copy()
#     random.shuffle(shuffled)
#     test  = shuffled[:test_n]
#     val   = shuffled[test_n:test_n + val_n]
#     train = shuffled[test_n + val_n:]
#     return train, val, test

# d_tr, d_val, d_te = split_category(DESERT_LOCS,  val_n=2, test_n=2)
# g_tr, g_val, g_te = split_category(GREENERY_LOCS, val_n=2, test_n=2)
# u_tr, u_val, u_te = split_category(URBAN_LOCS,    val_n=2, test_n=2)
# w_tr, w_val, w_te = split_category(WATER_LOCS,    val_n=2, test_n=2)

# TRAIN_LOCS = d_tr + g_tr + u_tr + w_tr
# VAL_LOCS   = d_val + g_val + u_val + w_val   # 8 locs, balanced types
# TEST_LOCS  = d_te + g_te + u_te + w_te       # 8 locs, balanced types

# print(f"Train: {len(TRAIN_LOCS)}, Val: {len(VAL_LOCS)}, Test: {len(TEST_LOCS)}")

# %%
def compute_indices(bands):
    B2 = bands[1].astype(float)   
    B3 = bands[2].astype(float)   
    B4 = bands[3].astype(float)   
    B8 = bands[7].astype(float)   
    B8A = bands[8].astype(float)   
    B11 = bands[10].astype(float)  
    e = 1e-10  

    # vegetation index: high for green areas, low for cities and water
    NDVI = (B8 - B4) / (B8 + B4 + e)

    # detects water stress in plants
    MOISTURE = (B8A - B11) / (B8A + B11 + e)

    # water index: high for water bodies
    NDWI = (B3 - B8) / (B3 + B8 + e)

    # built-up index: high for urban/cement areas
    NDBI = (B11 - B8) / (B11 + B8 + e)

    # bare soil index: helps separate sand from cement
    BSI = ((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2) + e)

    return np.stack([NDVI, MOISTURE, NDWI, NDBI, BSI], axis=0)


def load_image_and_mask(location_name):
    img_path = os.path.join(DATA_FOLDER, location_name + '_12band.tif')
    mask_path = os.path.join(DATA_FOLDER, location_name + '_mask_ESRI.tif')

    with rasterio.open(img_path) as f:
        bands = f.read().astype(np.float32)

    with rasterio.open(mask_path) as f:
        mask = f.read(1)

    # make sure image and mask have same size as GEE sometimes exports slightly different
    H = min(bands.shape[1], mask.shape[0])
    W = min(bands.shape[2], mask.shape[1])
    bands = bands[:, :H, :W]
    mask = mask[:H, :W]

    # normalize bands: Sentinel-2 values go from 0 to 10000
    # we divide by 10000 so all values are between 0 and 1
    bands = bands / 10000.0
    bands = np.clip(bands, 0, 1)

    # replace any NaN or invalid values with 0
    # NaN values come from cloud-masked pixels in the GEE export
    bands = np.nan_to_num(bands, nan=0.0, posinf=0.0, neginf=0.0)

    # compute the 5 spectral indices
    indices = compute_indices(bands)
    indices = np.nan_to_num(indices, nan=0.0, posinf=0.0, neginf=0.0)

    # combine raw bands + indices = 17 features per pixel
    features = np.concatenate([bands, indices], axis=0)
    return features, mask, bands


def get_pixels_from_locations(location_list):
    all_X = []
    all_y = []

    for loc in location_list:
        try:
            features, mask, bands = load_image_and_mask(loc)

            # reshape from (17, H, W) to (H*W, 17) so each row is one pixel
            num_features = features.shape[0]
            X = features.reshape(num_features, -1).T
            y = mask.flatten()

            # keep only labeled pixels :remove unknown class (0) and border areas
            labeled = y > 0
            X = X[labeled]
            y = y[labeled]

            # remove any pixel that still has NaN in its features
            clean = ~np.isnan(X).any(axis=1)
            all_X.append(X[clean])
            all_y.append(y[clean])

        except Exception as e:
            print(f'skipping {loc}: {e}')

    return np.vstack(all_X), np.concatenate(all_y)

# %%
print('loading training data')
X_train, y_train = get_pixels_from_locations(TRAIN_LOCS)

print('loading validation data')
X_val, y_val = get_pixels_from_locations(VAL_LOCS)

print('loading test data')
X_test, y_test = get_pixels_from_locations(TEST_LOCS)

print(f'\nTrain : {X_train.shape[0]:,} pixels, {X_train.shape[1]} features each')
print(f'Val : {X_val.shape[0]:,} pixels')
print(f'Test : {X_test.shape[0]:,} pixels')

print('\nclass distribution in training set:')
for label, name in zip(CLASS_LABELS, CLASS_NAMES):
    count = (y_train == label).sum()
    print(f'{name}: {count:,} pixels ({100*count/len(y_train):.1f}%)')

# %%
def sample_equal_per_class(X, y, n=600000):
    X_out = []
    y_out = []
    for label, name in zip(CLASS_LABELS, CLASS_NAMES):
        idx = np.where(y == label)[0]
        take = min(len(idx), n)
        chosen = np.random.RandomState(42).choice(idx, take, replace=False)
        X_out.append(X[chosen])
        y_out.append(y[chosen])
        print(f'  {name}: {take:,} pixels')
    return np.vstack(X_out), np.concatenate(y_out)

print('sampling balanced training set...')
X_tr, y_tr = sample_equal_per_class(X_train, y_train, n=200000)
print(f'total training pixels: {len(X_tr):,}')

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)   
X_val_s = scaler.transform(X_val)     
X_test_s = scaler.transform(X_test)

X_tr_s = np.nan_to_num(X_tr_s, nan=0.0)
X_val_s = np.nan_to_num(X_val_s, nan=0.0)
X_test_s = np.nan_to_num(X_test_s, nan=0.0)

assert not np.isnan(X_tr_s).any(),'NaN in training data'
assert not np.isnan(X_val_s).any(),'NaN in validation data'
assert not np.isnan(X_test_s).any(),'NaN in test data'
print('normalization done')

# %%
# IoU per class = TP / (TP + FP + FN)
def compute_miou(true_labels, predicted_labels):
    iou_scores = {}
    for i in range(len(CLASS_LABELS)):
        c = CLASS_LABELS[i]
        tp = 0
        fp = 0
        fn = 0
        
        for j in range(len(true_labels)):
            if predicted_labels[j] == c and true_labels[j] == c:
                tp += 1
            elif predicted_labels[j] == c and true_labels[j] != c:
                fp += 1
            elif predicted_labels[j] != c and true_labels[j] == c:
                fn += 1
        iou_scores[c] = tp / (tp + fp + fn + 1e-10)
    miou = sum(iou_scores.values()) / len(iou_scores)
    return iou_scores, miou

# %%
class MinimumDistanceClassifier:
    def __init__(self):
        self.signatures = {}   

    def train(self, X, y):
        for label in np.unique(y):
            class_pixels = X[y == label]
            self.signatures[label] = class_pixels.mean(axis=0)
        return self
    
    ## old version take more time 
    # def classify(self, X):
    #     class_labels = list(self.signatures.keys())
    #     predictions  = np.zeros(len(X), dtype=int)

    #     # process in batches to avoid memory issues
    #     batch_size = 50000
    #     start = 0

    #     while start < len(X):
    #         end   = min(start + batch_size, len(X))
    #         batch = X[start:end]

    #         # for each pixel in this batch find the nearest class
    #         for pixel_idx in range(len(batch)):
    #             pixel = batch[pixel_idx]

    #             min_dist   = float('inf')
    #             best_class = class_labels[0]

    #             # compute distance to each class signature
    #             for c in class_labels:
    #                 signature = self.signatures[c]

    #                 # euclidean distance
    #                 diff = pixel - signature
    #                 dist = np.sqrt(np.sum(diff * diff))

    #                 if dist < min_dist:
    #                     min_dist   = dist
    #                     best_class = c
    #             predictions[start + pixel_idx] = best_class
    #         start = end
    #     return predictions

    def classify(self, X):
        labels = np.array(list(self.signatures.keys()))
        means = np.array(list(self.signatures.values()))

        batch_size = 50000
        predictions = []

        for start in range(0, len(X), batch_size):
            batch = X[start : start + batch_size]
            differences = batch[:, np.newaxis, :] - means[np.newaxis, :, :]
            distances = np.sqrt((differences ** 2).sum(axis=2))
            closest = np.argmin(distances, axis=1)
            predictions.append(labels[closest])

        return np.concatenate(predictions)

# train the classifier
print('training minimum distance classifier')
mdc = MinimumDistanceClassifier()
mdc.train(X_tr_s, y_tr)

print('\nclass signatures (mean of spectral indices after normalization):')
index_names = ['NDVI', 'Moisture', 'NDWI', 'NDBI', 'BSI']
print(f"  {'class':<12}", '  '.join(f'{n:>9}' for n in index_names))
for label, name in zip(CLASS_LABELS, CLASS_NAMES):
    sig = mdc.signatures[label][-5:]
    print(f'  {name:<12}', '  '.join(f'{v:>9.4f}' for v in sig))

joblib.dump(mdc, os.path.join(MODEL_FOLDER, 'mdc_model.pkl'))
joblib.dump(scaler, os.path.join(MODEL_FOLDER, 'scaler.pkl'))
print('\nmodel saved.')

# %%
def run_evaluation(split_name, X, y):
    print(f'\n {split_name} results ')

    pred = mdc.classify(X)
    cm = confusion_matrix(y, pred, labels=CLASS_LABELS)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, cmap='Blues')
    ax.set_title(f'Error Matrix : {split_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_FOLDER, f'error_matrix_{split_name}.png'), dpi=150)
    plt.show()

    print('errors of omission (how many true pixels were missed):')
    for i, name in enumerate(CLASS_NAMES):
        err = 1 - cm[i, i] / cm[i].sum()
        print(f'  {name}: {err:.4f}')

    print('errors of commission (how many predicted pixels were wrong):')
    for i, name in enumerate(CLASS_NAMES):
        err = 1 - cm[i, i] / cm[:, i].sum()
        print(f'  {name}: {err:.4f}')

    # mIoU score
    iou_dict, miou = compute_miou(y, pred)
    print(f'\nmIoU: {miou:.4f}')
    for label, name in zip(CLASS_LABELS, CLASS_NAMES):
        print(f'  {name}: {iou_dict[label]:.4f}')

    print(classification_report(y, pred, target_names=CLASS_NAMES, labels=CLASS_LABELS))
    return miou, iou_dict

val_miou, val_iou = run_evaluation('Validation', X_val_s, y_val)
test_miou, test_iou = run_evaluation('Test', X_test_s, y_test)

# %%
def make_thematic_map(location_name):
    features, mask, bands = load_image_and_mask(location_name)
    H, W = mask.shape

    # flatten to pixels, scale, classify, then reshape back to image
    X_img = features.reshape(17, -1).T
    X_scaled= scaler.transform(X_img)
    X_scaled= np.nan_to_num(X_scaled, nan=0.0)
    pred_img= mdc.classify(X_scaled).reshape(H, W)

    # unknown border pixels stay as 0
    pred_img[mask == 0] = 0

    colors = ['black', 'darkgreen', 'yellow', 'darkblue', 'gray']
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm([0, 1, 2, 3, 4, 5], cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(mask, cmap=cmap, norm=norm)
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')

    im = axes[1].imshow(pred_img, cmap=cmap, norm=norm)
    axes[1].set_title('MDC Prediction')
    axes[1].axis('off')

    cbar = plt.colorbar(im, ax=axes, ticks=[0.5, 1.5, 2.5, 3.5, 4.5],
                        fraction=0.02, pad=0.04)
    cbar.ax.set_yticklabels(['Unknown', 'Greenery', 'Sand', 'Water', 'Cement'])

    plt.suptitle(f'Land Cover Map — {location_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_FOLDER, f'{location_name}_prediction.png'), dpi=150)
    plt.show()

for loc in TEST_LOCS[:3]:
    make_thematic_map(loc)

# %% [markdown]
# ---
# # Part 2 — Deep Learning: U-Net Semantic Segmentation
# 
# We now implement a **U-Net** — a fully-convolutional encoder-decoder architecture originally designed for biomedical image segmentation but proven highly effective for satellite land-cover mapping.  
# The network takes a **(17, H, W)** tensor (12 bands + 5 spectral indices) and produces a **(4, H, W)** class-probability map.
# 
# Key design choices:
# - **Patch-based training** (128×128 tiles) — avoids GPU OOM on full-scene tensors  
# - **Skip connections** — preserve fine spatial detail that deep pooling erases  
# - **Class-weighted CrossEntropy** — compensates for Sand/Water pixel imbalance  
# - **Unknown-class mask** — class-0 pixels are excluded from loss and metrics  

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

# empty cash
if torch.cuda.is_available():
    torch.cuda.empty_cache()


from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# %%
# def compute_miou(true_labels, predicted_labels):
#     iou_scores = {}
#     true_labels = np.asarray(true_labels)
#     predicted_labels = np.asarray(predicted_labels)
#     for c in CLASS_LABELS:
#         pred_c   = predicted_labels == c
#         target_c = true_labels == c
#         intersection = (pred_c & target_c).sum()
#         union        = (pred_c | target_c).sum()
#         iou_scores[c] = float(intersection / (union + 1e-10)) if union > 0 else 0.0
#     miou = sum(iou_scores.values()) / len(iou_scores)
#     return iou_scores, miou

# %% [markdown]
# ## 2.1 — Patch Dataset
# 
# We slice each full scene into **128×128 patches** at training time.  
# Patches where the valid (non-zero) pixel ratio is below **15 %** are discarded — they are mostly border artefacts from the GEE export and would confuse the network.

# %%
PATCH_SIZE   = 128
PATCH_STRIDE = 96    # overlap keeps border context
MIN_VALID    = 0.15  # discard patches with < 15 % labeled pixels
IN_CHANNELS  = 17
NUM_CLASSES  = 5     # 0-Unknown, 1-Greenery, 2-Sand, 3-Water, 4-Cement


# %%

# class PatchDataset(Dataset):
#     """
#     Slices every scene in `location_list` into (PATCH_SIZE × PATCH_SIZE) patches.
#     Patches dominated by unknown pixels are skipped.
#     """
#     def __init__(self, location_list, augment=False):
#         self.patches = []    # list of (features_patch, mask_patch)
#         self.augment = augment
#         self._build(location_list)

#     def _build(self, location_list):
#         for loc in location_list:
#             try:
#                 features, mask, bands = load_image_and_mask(loc)  # reuse function from Part 1
#                 H, W = mask.shape
#                 for r in range(0, H - PATCH_SIZE + 1, PATCH_STRIDE):
#                     for c in range(0, W - PATCH_SIZE + 1, PATCH_STRIDE):
#                         m_patch = mask[r:r+PATCH_SIZE, c:c+PATCH_SIZE]
#                         valid_ratio = (m_patch > 0).mean()
#                         if valid_ratio < MIN_VALID:
#                             continue
#                         f_patch = features[:, r:r+PATCH_SIZE, c:c+PATCH_SIZE]
#                         self.patches.append((
#                             f_patch.astype(np.float32),
#                             m_patch.astype(np.int64)
#                         ))
#             except Exception as e:
#                 print(f'  skipping {loc}: {e}')
#         print(f'  built {len(self.patches):,} patches from {len(location_list)} scenes')

#     def __len__(self):
#         return len(self.patches)

#     def __getitem__(self, idx):
#         x, y = self.patches[idx]
#         x = torch.from_numpy(x)
#         y = torch.from_numpy(y)
#         if self.augment:
#             # random horizontal / vertical flip — free augmentation for satellite images
#             if torch.rand(1) > 0.5:
#                 x = torch.flip(x, dims=[2])
#                 y = torch.flip(y, dims=[1])
#             if torch.rand(1) > 0.5:
#                 x = torch.flip(x, dims=[1])
#                 y = torch.flip(y, dims=[0])
#             # random 90° rotation
#             k = torch.randint(0, 4, (1,)).item()
#             x = torch.rot90(x, k, dims=[1, 2])
#             y = torch.rot90(y, k, dims=[0, 1])
#         return x, y


# %%
class PatchDataset(Dataset):
    def __init__(self, location_list, augment=False):
        self.patches = []
        self.augment = augment
        self._build(location_list)

    def _build(self, location_list):
        for loc in location_list:
            try:
                features, mask, bands = load_image_and_mask(loc)
                H, W = mask.shape
                hH, hW = H // 2, W // 2

                # Split into 4 quadrants, treat each as a new scene
                quadrants = [
                    (0,   hH,  0,  hW),   # top-left
                    (0,   hH, hW,   W),   # top-right
                    (hH,   H,  0,  hW),   # bottom-left
                    (hH,   H, hW,   W),   # bottom-right
                ]

                for r0, r1, c0, c1 in quadrants:
                    f_quad = features[:, r0:r1, c0:c1]   # (17, hH, hW)
                    m_quad = mask[r0:r1, c0:c1]           # (hH, hW)

                    # Skip quadrant if almost entirely unlabeled
                    if (m_quad > 0).mean() < MIN_VALID:
                        continue

                    # Sliding-window patch extraction on this quadrant
                    qH, qW = m_quad.shape
                    for r in range(0, qH - PATCH_SIZE + 1, PATCH_STRIDE):
                        for c in range(0, qW - PATCH_SIZE + 1, PATCH_STRIDE):
                            m_patch = m_quad[r:r+PATCH_SIZE, c:c+PATCH_SIZE]
                            if (m_patch > 0).mean() < MIN_VALID:
                                continue
                            f_patch = f_quad[:, r:r+PATCH_SIZE, c:c+PATCH_SIZE]
                            self.patches.append((
                                f_patch.astype(np.float32),
                                m_patch.astype(np.int64)
                            ))

            except Exception as e:
                print(f'  skipping {loc}: {e}')

        print(f'  built {len(self.patches):,} patches from {len(location_list)} scenes')

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        x, y = self.patches[idx]
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        if self.augment:
            # random horizontal / vertical flip
            if torch.rand(1) > 0.5:
                x = torch.flip(x, dims=[2])
                y = torch.flip(y, dims=[1])
            if torch.rand(1) > 0.5:
                x = torch.flip(x, dims=[1])
                y = torch.flip(y, dims=[0])
            # random 90° rotation
            k = torch.randint(0, 4, (1,)).item()
            x = torch.rot90(x, k, dims=[1, 2])
            y = torch.rot90(y, k, dims=[0, 1])
        return x, y

# %%
# # load the dataset and create dataloaders
# train_ds = PatchDataset(TRAIN_LOCS, augment=True)
# val_ds   = PatchDataset(VAL_LOCS,   augment=False)
# test_ds  = PatchDataset(TEST_LOCS,  augment=False)
# train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
# val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
# test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

print('Building datasets ...')
train_dataset = PatchDataset(TRAIN_LOCS, augment=True)   # ← augmentation ON
val_dataset   = PatchDataset(VAL_LOCS,   augment=False)
test_dataset  = PatchDataset(TEST_LOCS,  augment=False)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=8, shuffle=False, num_workers=0)

# %% [markdown]
# ## 2.2 — U-Net Architecture
# 
# ```
# Input (17 ch)
#   │
#   ▼  Encoder
#   E1 (64)  ──────────────────────────────────── skip ──┐
#     MaxPool                                             │
#   E2 (128) ─────────────────────────────── skip ──┐    │
#     MaxPool                                        │    │
#   E3 (256) ────────────────────── skip ──┐         │    │
#     MaxPool                              │         │    │
#   E4 (512) ─────────── skip ──┐          │         │    │
#     MaxPool                   │          │         │    │
#   Bottleneck (1024)           │          │         │    │
#   Decoder                     │          │         │    │
#   D4 (512) ◄──────────────────┘          │         │    │
#   D3 (256) ◄───────────────────────────-┘         │    │
#   D2 (128) ◄──────────────────────────────────────┘    │
#   D1 (64)  ◄─────────────────────────────────────────--┘
#   Output (NUM_CLASSES)
# ```

# %%
def double_conv(in_ch, out_ch):
    """Two successive 3×3 convolutions with BatchNorm and ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels=17, num_classes=5, base_filters=64):
        super().__init__()
        f = base_filters

        # ── Encoder ──────────────────────────────────────
        self.enc1 = double_conv(in_channels, f)        # → (f,   H,   W)
        self.enc2 = double_conv(f,     f * 2)          # → (2f,  H/2, W/2)
        self.enc3 = double_conv(f * 2, f * 4)          # → (4f,  H/4, W/4)
        self.enc4 = double_conv(f * 4, f * 8)          # → (8f,  H/8, W/8)

        self.pool = nn.MaxPool2d(2)

        # ── Bottleneck ───────────────────────────────────
        self.bottleneck = double_conv(f * 8, f * 16)   # → (16f, H/16, W/16)

        # ── Decoder ──────────────────────────────────────
        self.up4   = nn.ConvTranspose2d(f * 16, f * 8,  kernel_size=2, stride=2)
        self.dec4  = double_conv(f * 16, f * 8)

        self.up3   = nn.ConvTranspose2d(f * 8,  f * 4,  kernel_size=2, stride=2)
        self.dec3  = double_conv(f * 8,  f * 4)

        self.up2   = nn.ConvTranspose2d(f * 4,  f * 2,  kernel_size=2, stride=2)
        self.dec2  = double_conv(f * 4,  f * 2)

        self.up1   = nn.ConvTranspose2d(f * 2,  f,      kernel_size=2, stride=2)
        self.dec1  = double_conv(f * 2,  f)

        # ── Output head ──────────────────────────────────
        self.out_conv = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        # Bottleneck
        bn = self.bottleneck(self.pool(s4))

        # Decoder — upsample then concatenate the skip connection
        x = self.dec4(torch.cat([self.up4(bn), s4], dim=1))
        x = self.dec3(torch.cat([self.up3(x),  s3], dim=1))
        x = self.dec2(torch.cat([self.up2(x),  s2], dim=1))
        x = self.dec1(torch.cat([self.up1(x),  s1], dim=1))

        return self.out_conv(x)   # raw logits: (B, NUM_CLASSES, H, W)


# %%
# the model definition is the same as before, but now we create an instance and move it to the GPU if available
model = UNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, base_filters=64).to(DEVICE)
print(model)

# %% [markdown]
# ## 2.3 — Loss Function & Optimizer
# 
# **Class weights** correct for imbalanced pixel counts (desert Sand dominates Egyptian scenes).  
# **Unknown (class 0)** is excluded by setting its weight to 0 and using `ignore_index=0`.

# %%
# Estimate class frequencies from the training pixel set y_train (Part-1 variable)
class_counts = np.array([(y_train == c).sum() for c in range(NUM_CLASSES)], dtype=np.float64)
class_counts[0] = 1.0   # prevent div-by-zero; weight will be set to 0 anyway

# Inverse-frequency weights; Unknown class gets 0
weights = 1.0 / class_counts
weights[0] = 0.0

# apply a manual correction multiplier per class
# based on what the confusion matrices tell us is hard
# correction = np.array([
    # 0.0,   # Unknown — always 0
    # 1.5,   # Greenery — boost: 173k confused with Sand, 65k with Cement
    # 0.8,   # Sand — reduce: already excellent (IoU 0.724), dominant class
    # 2.0,   # Water — boost most: still confused with Cement (75k pixels)
    # 1.2,   # Cement — small boost: good IoU but Greenery→Cement confusion hurts
# ])
correction = np.array(np.ones(NUM_CLASSES))
weights = weights * correction

weights = weights / weights[1:].sum()  # normalise over the 4 active classes
weights_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

print('class weights:')
all_names = ['Unknown', 'Greenery', 'Sand', 'Water', 'Cement']
for n, w in zip(all_names, weights):
    print(f'  {n:<12}: {w:.6f}')

criterion = nn.CrossEntropyLoss(weight=weights_tensor, ignore_index=0)

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

# Cosine annealing — gently reduces LR towards zero over training
NUM_EPOCHS = 30
scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# %% [markdown]
# ## 2.4 — Training Loop

# %%
def compute_patch_miou(preds, targets):
    """
    Vectorised mIoU over a batch of predicted and target tensors.
    Unknown (class 0) is excluded.
    preds  : (N,) int64 flattened predictions
    targets: (N,) int64 flattened ground truth
    """
    iou_list = []
    for c in CLASS_LABELS:            # CLASS_LABELS = [1,2,3,4] 
        pred_c   = (preds   == c)
        target_c = (targets == c)
        intersection = (pred_c & target_c).sum()
        union        = (pred_c | target_c).sum()
        if union == 0:
            continue
        iou_list.append(intersection / union)
    return float(np.mean(iou_list)) if iou_list else 0.0


def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, total_miou, n_batches = 0.0, 0.0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            logits = model(X_batch)           # (B, 5, 128, 128)
            loss   = criterion(logits, y_batch)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            preds   = logits.argmax(dim=1).cpu().numpy().flatten()
            targets = y_batch.cpu().numpy().flatten()

            # exclude unknown pixels from metric
            valid = targets > 0
            batch_miou = compute_patch_miou(preds[valid], targets[valid])

            total_loss  += loss.item()
            total_miou  += batch_miou
            n_batches   += 1

    return total_loss / n_batches, total_miou / n_batches

# empty cache after each epoch to prevent OOM
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    
BEST_MODEL_PATH = os.path.join(MODEL_FOLDER, 'unet_best.pth')

history = {'train_loss': [], 'val_loss': [], 'train_miou': [], 'val_miou': []}
best_val_miou = 0.0

print(f'Training U-Net for {NUM_EPOCHS} epochs on {DEVICE} …')
print(f'{"Epoch":>5}  {"Train Loss":>10}  {"Val Loss":>10}  {"Train mIoU":>11}  {"Val mIoU":>10}')

for epoch in range(1, NUM_EPOCHS + 1):
    tr_loss, tr_miou = run_epoch(train_loader, train=True)
    vl_loss, vl_miou = run_epoch(val_loader,   train=False)
    scheduler.step()

    history['train_loss'].append(tr_loss)
    history['val_loss'].append(vl_loss)
    history['train_miou'].append(tr_miou)
    history['val_miou'].append(vl_miou)

    flag = ''
    if vl_miou > best_val_miou:
        best_val_miou = vl_miou
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        flag = '  ← best'

    print(f'{epoch:>5}  {tr_loss:>10.4f}  {vl_loss:>10.4f}  {tr_miou:>11.4f}  {vl_miou:>10.4f}{flag}')
    # empty cache after each epoch to prevent OOM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f'\nBest val mIoU: {best_val_miou:.4f}')

# %% [markdown]
# ## 2.5 — Training Curves

# %%
epochs = range(1, NUM_EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(epochs, history['train_loss'], label='Train')
axes[0].plot(epochs, history['val_loss'],   label='Val')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Cross-Entropy Loss')
axes[0].set_title('Training & Validation Loss'); axes[0].legend()

axes[1].plot(epochs, history['train_miou'], label='Train')
axes[1].plot(epochs, history['val_miou'],   label='Val')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('mIoU')
axes[1].set_title('Training & Validation mIoU'); axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(MODEL_FOLDER, 'unet_training_curves.png'), dpi=150)
plt.show()

# %% [markdown]
# ## 2.6 — Full Evaluation on Test Set
# 
# We reload the best checkpoint and run pixel-level evaluation across all test scenes.

# %%
def evaluate_unet(loader, split_name):
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch.to(DEVICE))
            preds  = logits.argmax(dim=1).cpu().numpy().flatten()
            targets = y_batch.numpy().flatten()

            valid = targets > 0
            all_preds.append(preds[valid])
            all_targets.append(targets[valid])

    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Error matrix
    cm = confusion_matrix(all_targets, all_preds, labels=CLASS_LABELS)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, cmap='Greens')
    ax.set_title(f'U-Net Error Matrix: {split_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_FOLDER, f'unet_error_matrix_{split_name}.png'), dpi=150)
    plt.show()

    print(f'Errors of omission (missed true pixels):')
    for i, name in enumerate(CLASS_NAMES):
        err = 1 - cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f'  {name}: {err:.4f}')

    print(f'\nErrors of commission (false positives):')
    for i, name in enumerate(CLASS_NAMES):
        err = 1 - cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
        print(f'  {name}: {err:.4f}')

    iou_dict, miou = compute_miou(all_targets, all_preds)   # reuse Part-1 function
    print(f'\nmIoU: {miou:.4f}')
    for label, name in zip(CLASS_LABELS, CLASS_NAMES):
        print(f'  {name}: {iou_dict[label]:.4f}')

    print(classification_report(all_targets, all_preds, target_names=CLASS_NAMES, labels=CLASS_LABELS))
    return miou, iou_dict


print('=== Validation set ===')
unet_val_miou, unet_val_iou = evaluate_unet(val_loader, 'Validation')

print('\n=== Test set ===')
unet_test_miou, unet_test_iou = evaluate_unet(test_loader, 'Test')

# %% [markdown]
# ## 2.7 — Thematic Maps: U-Net Predictions vs Ground Truth
# 
# We run the best checkpoint over the full-resolution test scenes (not patches) using a sliding-window inference to handle arbitrary image sizes without padding artefacts.

# %%
def sliding_window_predict(features_np, patch_size=128, stride=64):
    """
    Run the U-Net over a full scene using a sliding window.
    Overlapping predictions are averaged before argmax.
    features_np : (17, H, W) float32
    """
    model.eval()
    _, H, W = features_np.shape
    logit_sum   = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
    count_map   = np.zeros((H, W),              dtype=np.float32)

    with torch.no_grad():
        for r in range(0, H - patch_size + 1, stride):
            for c in range(0, W - patch_size + 1, stride):
                patch = features_np[:, r:r+patch_size, c:c+patch_size]
                t     = torch.from_numpy(patch).unsqueeze(0).to(DEVICE)   # (1,17,P,P)
                logit = model(t).squeeze(0).cpu().numpy()                  # (5,P,P)
                logit_sum[:, r:r+patch_size, c:c+patch_size] += logit
                count_map[r:r+patch_size, c:c+patch_size]    += 1.0

    count_map = np.maximum(count_map, 1)                # avoid div-by-zero at borders
    avg_logit = logit_sum / count_map[np.newaxis, :, :]
    return avg_logit.argmax(axis=0)                     # (H, W) class map


def make_unet_thematic_map(location_name):
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))

    features, mask, bands = load_image_and_mask(location_name)
    pred_img = sliding_window_predict(features.astype(np.float32), PATCH_SIZE)

    # keep unknown border pixels as 0
    pred_img[mask == 0] = 0
    # after sliding_window_predict(), apply NDWI override
    # ndwi = (bands[2] - bands[7]) / (bands[2] + bands[7] + 1e-10)  # B3-B8
    # pred_img[ndwi > 0.20] = 3  # force Water class where NDWI is clearly positive
    # pred_img[mask == 0] = 0     # restore border mask
# 
    colors = ['black', 'darkgreen', 'yellow', 'darkblue', 'gray']
    cmap   = mcolors.ListedColormap(colors)
    norm   = mcolors.BoundaryNorm([0, 1, 2, 3, 4, 5], cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(mask,     cmap=cmap, norm=norm)
    axes[0].set_title('Ground Truth');     axes[0].axis('off')

    im = axes[1].imshow(pred_img, cmap=cmap, norm=norm)
    axes[1].set_title('U-Net Prediction'); axes[1].axis('off')

    cbar = plt.colorbar(im, ax=axes, ticks=[0.5,1.5,2.5,3.5,4.5],
                        fraction=0.02, pad=0.04)
    cbar.ax.set_yticklabels(['Unknown','Greenery','Sand','Water','Cement'])

    plt.suptitle(f'U-Net Land Cover Map — {location_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_FOLDER, f'{location_name}_unet_prediction.png'), dpi=150)
    plt.show()

for loc in TEST_LOCS[:3]:
    make_unet_thematic_map(loc)

# %% [markdown]
# ---
# # Part 3 — Discussion & Comparative Analysis
# 
# This section critically compares the **Minimum Distance Classifier (MDC)** against the **U-Net** across four axes: performance, computation, design reasoning, and edge cases.

# %%
# ── Summary table ──────────────────────────────────────────────────────────────
print('=' * 65)
print(f'{"Metric":<30} {"MDC":>15} {"U-Net":>15}')
print('=' * 65)

rows = [
    ('Val  mIoU',       f'{val_miou:.4f}',       f'{unet_val_miou:.4f}'),
    ('Test mIoU',       f'{test_miou:.4f}',       f'{unet_test_miou:.4f}'),
]
for label, name in zip(CLASS_LABELS, CLASS_NAMES):
    rows.append((f'  Test IoU — {name}',
                 f'{test_iou[label]:.4f}',
                 f'{unet_test_iou[label]:.4f}'))

rows += [
    # ('Trainable parameters',  '0 (no gradient)', f'{total_params:,}'),
    ('Spatial context used',  'None (per-pixel)',  f'{PATCH_SIZE}×{PATCH_SIZE} patches'),
]

for r in rows:
    print(f'{r[0]:<30} {r[1]:>15} {r[2]:>15}')
print('=' * 65)

# =================================================================
# Metric                                     MDC           U-Net
# =================================================================
# Val  mIoU                               0.6232          0.6674
# Test mIoU                               0.4965          0.6587
#   Test IoU — Greenery                   0.4569          0.5399
#   Test IoU — Sand                       0.4918          0.7242
#   Test IoU — Water                      0.6280          0.6451
#   Test IoU — Cement                     0.4092          0.7255
# Trainable parameters           0 (no gradient)       7,767,205
# Spatial context used           None (per-pixel) 128×128 patches
# =================================================================
# =================================================================
# Metric                                     MDC           U-Net
# =================================================================
# Val  mIoU                               0.5753          0.6893
# Test mIoU                               0.6199          0.6635
#   Test IoU — Greenery                   0.5153          0.5727
#   Test IoU — Sand                       0.7352          0.7261
#   Test IoU — Water                      0.7960          0.8254
#   Test IoU — Cement                     0.4333          0.5299
# Trainable parameters           0 (no gradient)       7,767,205
# Spatial context used           None (per-pixel) 128×128 patches
# =================================================================
# =================================================================
# Metric                                     MDC           U-Net
# =================================================================
# Val  mIoU                               0.5753          0.6788
# Test mIoU                               0.6199          0.6660
#   Test IoU — Greenery                   0.5153          0.5696
#   Test IoU — Sand                       0.7352          0.7282
#   Test IoU — Water                      0.7960          0.8248
#   Test IoU — Cement                     0.4333          0.5415
# Trainable parameters           0 (no gradient)       7,767,205
# Spatial context used           None (per-pixel) 128×128 patches
# =================================================================
# =================================================================
# Metric                                     MDC           U-Net
# =================================================================
# Val  mIoU                               0.6232          0.6522
# Test mIoU                               0.4965          0.5573
#   Test IoU — Greenery                   0.4569          0.5563
#   Test IoU — Sand                       0.4918          0.4299
#   Test IoU — Water                      0.6280          0.6497
#   Test IoU — Cement                     0.4092          0.5931
# Trainable parameters           0 (no gradient)       7,767,205
# Spatial context used           None (per-pixel) 128×128 patches
# =================================================================
# =================================================================
# Metric                                     MDC           U-Net
# =================================================================
# Val  mIoU                               0.6232          0.6760
# Test mIoU                               0.4965          0.6448
#   Test IoU — Greenery                   0.4569          0.5393
#   Test IoU — Sand                       0.4918          0.6625
#   Test IoU — Water                      0.6280          0.6476
#   Test IoU — Cement                     0.4092          0.7297
# Trainable parameters           0 (no gradient)      31,045,957
# Spatial context used           None (per-pixel) 128×128 patches
# =================================================================

# %%
# ── Visual comparison bar chart ─────────────────────────────────────────────
x = np.arange(len(CLASS_NAMES))
width = 0.35

mdc_iou_vals  = [test_iou[c]      for c in CLASS_LABELS]
unet_iou_vals = [unet_test_iou[c] for c in CLASS_LABELS]

fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - width/2, mdc_iou_vals,  width, label='MDC',   color='steelblue',  alpha=0.85)
bars2 = ax.bar(x + width/2, unet_iou_vals, width, label='U-Net', color='darkorange', alpha=0.85)

ax.axhline(test_miou,      color='steelblue',  linestyle='--', linewidth=1.2, label=f'MDC mIoU  {test_miou:.3f}')
ax.axhline(unet_test_miou, color='darkorange', linestyle='--', linewidth=1.2, label=f'U-Net mIoU {unet_test_miou:.3f}')

ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES)
ax.set_ylabel('IoU'); ax.set_ylim(0, 1)
ax.set_title('Per-class IoU: MDC vs U-Net (Test Set)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(MODEL_FOLDER, 'comparison_iou_barchart.png'), dpi=150)
plt.show()

# %%
# save the summary table as a text file
with open(os.path.join(MODEL_FOLDER, 'summary_table.txt'), 'w') as f:
    f.write('=' * 65 + '\n')
    f.write(f'{"Metric":<30} {"MDC":>15} {"U-Net":>15}\n')
    f.write('=' * 65 + '\n')
    for r in rows:
        f.write(f'{r[0]:<30} {r[1]:>15} {r[2]:>15}\n')
    f.write('=' * 65 + '\n')

# %% [markdown]
# ## 3.1 — Performance Analysis
# 
# ### Minimum Distance Classifier (MDC)
# 
# The MDC classifies each pixel purely by its Euclidean distance to 4 class mean signatures in 17-dimensional feature space. Despite its simplicity it performs competitively on spectrally distinct classes (e.g. **Water**, which has a highly characteristic NDWI profile) and is deterministic and interpretable.
# 
# Its main weakness is **lack of spatial context**: a single bright pixel surrounded by sand is just as likely to be labelled 'Cement' as an entire city block, because the classifier cannot see its neighbourhood. This leads to salt-and-pepper noise in the predicted maps, especially in transitional zones (e.g. irrigated desert margins between Sand and Greenery).
# 
# **Computational profile** — training is instant (just computing 4 mean vectors). Inference is a vectorised distance computation: O(N × C × F) where N is pixel count, C = 4 classes, F = 17 features.
# 
# ### U-Net
# 
# The U-Net sees each pixel in the context of a 128×128 receptive field. Skip connections feed high-resolution spatial detail from the encoder directly into the decoder, allowing the network to simultaneously reason at coarse semantic level (is this region urban or desert?) and fine detail (where exactly does the urban boundary lie?).
# 
# Expected gains over MDC:
# - **Cement/Greenery boundary precision** — skip connections recover sharp edges lost in pooling.
# - **Mixed-class transitional zones** — spatial context resolves ambiguous pixels.
# - **Robustness to single-pixel noise** — a spatially smooth predicted map is enforced by the convolutional receptive field.
# 
# The cost is ~20-40 minutes of GPU training, a 31M-parameter model, and more complex hyperparameter tuning.
# 
# ## 3.2 — Architectural Reasoning
# 
# | Design choice | Reasoning |
# |---|---|
# | 4 encoder levels (f→2f→4f→8f) | Captures features from 3-pixel texture up to 128-pixel spatial patterns |
# | Skip connections | Preserve fine boundaries; critical for thin linear features like canal banks |
# | BatchNorm + ReLU after every conv | Stabilises training across the wide spectral dynamic range of Sentinel-2 |
# | AdamW + cosine annealing | AdamW weight-decay prevents overfitting on small datasets; cosine schedule avoids abrupt LR drops |
# | ignore_index=0 in CrossEntropy | Prevents the network from being penalised for Unknown-class boundary pixels |
# | Patch stride < patch size (96 < 128) | Overlapping patches at training time expose the network to more boundary configurations |
# | Sliding-window inference with averaging | Avoids border artefacts; overlapping logit averaging smooths predictions near patch edges |
# 
# ## 3.3 — Edge Cases & Failure Modes
# 
# **1. Spectrally ambiguous classes — Sand vs Cement.**  
# Dry concrete and bright desert sand share very similar reflectance in the visible bands. Both models struggle here. The MDC relies heavily on NDBI and BSI to separate them, but these indices also respond to rocky terrain. The U-Net has additional context (e.g. regular grid patterns of roads vs. smooth dune surfaces) but requires enough training data for these patterns to be learnt.
# 
# **2. Cloud-masked pixels (NaN → 0).**  
# Replacing NaN from cloud masking with 0 may introduce spurious low-reflectance pixels. A learnt imputation (e.g. filling from spatial neighbours) would be more principled but was avoided here to keep the pipeline simple and reproducible.
# 
# **3. Seasonal variability.**  
# The dataset is limited to May–August 2023 (low-cloud summer). Agricultural fields in the Nile Delta undergo dramatic NDVI changes between growing seasons. A model trained only on summer imagery may misclassify bare harvested fields as Sand in winter test images.
# 
# **4. Small water bodies.**  
# Narrow irrigation canals (< 10 m wide, i.e. sub-pixel at 10 m Sentinel-2 resolution) are inherently unresolvable. Neither model can detect them reliably — this is a sensor limitation rather than a modelling failure.
# 
# ## 3.4 — Conclusion
# 
# The MDC is an excellent **baseline**: zero training time, fully interpretable, and surprisingly effective on spectrally distinctive classes. It should be the first thing run on any new satellite dataset to establish a lower-bound reference.
# 
# The U-Net trades interpretability and speed for substantially higher mIoU, especially in spatially complex regions. For production land-cover mapping of Egyptian terrain — where urban expansion into the desert creates intricate Sand/Cement boundaries — the U-Net is the preferred model.
# 
# A practical deployment pipeline would use both: the MDC for rapid quality-control screening of new imagery, and the U-Net for final high-confidence maps.


