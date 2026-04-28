"""
Test the U-Net model on all scenes in the samples_prepared folder.
Produces:
  - Thematic map PNG (Ground Truth vs U-Net Prediction) per scene
  - Confusion matrix PNG per scene
  - Per-scene metrics printed to console
  - Overall mIoU summary across all scenes
"""

import os
import numpy as np
import torch
import torch.nn as nn
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLES_FOLDER = 'samples_prepared'
MODEL_FOLDER   = 'models'
BEST_MODEL_PATH = os.path.join(MODEL_FOLDER, 'unet_best.pth')
OUTPUT_FOLDER  = os.path.join(MODEL_FOLDER, 'samples_results')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

PATCH_SIZE   = 128
STRIDE       = 64
IN_CHANNELS  = 17
NUM_CLASSES  = 5     # 0-Unknown, 1-Greenery, 2-Sand, 3-Water, 4-Cement
CLASS_NAMES  = ['Greenery', 'Sand', 'Water', 'Cement']
CLASS_LABELS = [1, 2, 3, 4]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

# ── Spectral index computation (identical to notebook) ────────────────────────
def compute_indices(bands):
    B2  = bands[1].astype(float)
    B3  = bands[2].astype(float)
    B4  = bands[3].astype(float)
    B8  = bands[7].astype(float)
    B8A = bands[8].astype(float)
    B11 = bands[10].astype(float)
    e   = 1e-10

    NDVI     = (B8  - B4)  / (B8  + B4  + e)
    MOISTURE = (B8A - B11) / (B8A + B11 + e)
    NDWI     = (B3  - B8)  / (B3  + B8  + e)
    NDBI     = (B11 - B8)  / (B11 + B8  + e)
    BSI      = ((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2) + e)

    return np.stack([NDVI, MOISTURE, NDWI, NDBI, BSI], axis=0)


def load_sample(location_name):
    """
    Load from samples_prepared/<location_name>/<location_name>_Spectral.tif
                              and <location_name>_Mask.tif
    Returns: features (17, H, W), mask (H, W), bands (12, H, W)
    """
    spectral_path = os.path.join(SAMPLES_FOLDER, location_name,
                                 f'{location_name}_Spectral.tif')
    mask_path     = os.path.join(SAMPLES_FOLDER, location_name,
                                 f'{location_name}_Mask.tif')

    with rasterio.open(spectral_path) as f:
        bands = f.read().astype(np.float32)

    with rasterio.open(mask_path) as f:
        mask = f.read(1)

    # Align sizes (GEE sometimes exports slightly different dims)
    H = min(bands.shape[1], mask.shape[0])
    W = min(bands.shape[2], mask.shape[1])
    bands = bands[:, :H, :W]
    mask  = mask[:H, :W]

    # Normalise Sentinel-2 reflectance to [0, 1]
    bands = bands / 10000.0
    bands = np.clip(bands, 0, 1)
    bands = np.nan_to_num(bands, nan=0.0, posinf=0.0, neginf=0.0)

    indices  = compute_indices(bands)
    indices  = np.nan_to_num(indices, nan=0.0, posinf=0.0, neginf=0.0)
    features = np.concatenate([bands, indices], axis=0)   # (17, H, W)
    return features, mask, bands


# ── U-Net definition (identical to notebook) ─────────────────────────────────
def double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )

class UNet(nn.Module):
    def __init__(self, in_channels=17, num_classes=5, base_filters=64):
        super().__init__()
        f = base_filters
        self.enc1 = double_conv(in_channels, f)
        self.enc2 = double_conv(f,     f * 2)
        self.enc3 = double_conv(f * 2, f * 4)
        self.enc4 = double_conv(f * 4, f * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = double_conv(f * 8, f * 16)
        self.up4  = nn.ConvTranspose2d(f * 16, f * 8,  kernel_size=2, stride=2)
        self.dec4 = double_conv(f * 16, f * 8)
        self.up3  = nn.ConvTranspose2d(f * 8,  f * 4,  kernel_size=2, stride=2)
        self.dec3 = double_conv(f * 8,  f * 4)
        self.up2  = nn.ConvTranspose2d(f * 4,  f * 2,  kernel_size=2, stride=2)
        self.dec2 = double_conv(f * 4,  f * 2)
        self.up1  = nn.ConvTranspose2d(f * 2,  f,      kernel_size=2, stride=2)
        self.dec1 = double_conv(f * 2,  f)
        self.out_conv = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        bn = self.bottleneck(self.pool(s4))
        x  = self.dec4(torch.cat([self.up4(bn), s4], dim=1))
        x  = self.dec3(torch.cat([self.up3(x),  s3], dim=1))
        x  = self.dec2(torch.cat([self.up2(x),  s2], dim=1))
        x  = self.dec1(torch.cat([self.up1(x),  s1], dim=1))
        return self.out_conv(x)


# ── Load model ────────────────────────────────────────────────────────────────
model = UNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
model.eval()
print(f'Loaded model from {BEST_MODEL_PATH}')


# ── Sliding-window inference (identical to notebook) ─────────────────────────
def sliding_window_predict(features_np, patch_size=PATCH_SIZE, stride=STRIDE):
    _, H, W = features_np.shape
    logit_sum = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    with torch.no_grad():
        for r in range(0, H - patch_size + 1, stride):
            for c in range(0, W - patch_size + 1, stride):
                patch = features_np[:, r:r+patch_size, c:c+patch_size]
                t     = torch.from_numpy(patch).unsqueeze(0).to(DEVICE)
                logit = model(t).squeeze(0).cpu().numpy()
                logit_sum[:, r:r+patch_size, c:c+patch_size] += logit
                count_map[r:r+patch_size, c:c+patch_size]    += 1.0

    count_map = np.maximum(count_map, 1)
    avg_logit = logit_sum / count_map[np.newaxis, :, :]
    return avg_logit.argmax(axis=0)


# ── mIoU helper ──────────────────────────────────────────────────────────────
def compute_miou(true_labels, predicted_labels):
    iou_scores = {}
    for c in CLASS_LABELS:
        tp = ((predicted_labels == c) & (true_labels == c)).sum()
        fp = ((predicted_labels == c) & (true_labels != c)).sum()
        fn = ((predicted_labels != c) & (true_labels == c)).sum()
        iou_scores[c] = tp / (tp + fp + fn + 1e-10)
    miou = np.mean(list(iou_scores.values()))
    return iou_scores, miou


# ── Colour map ────────────────────────────────────────────────────────────────
colors = ['black', 'darkgreen', 'yellow', 'darkblue', 'gray']
cmap   = mcolors.ListedColormap(colors)
norm   = mcolors.BoundaryNorm([0, 1, 2, 3, 4, 5], cmap.N)


# ── Discover scenes ───────────────────────────────────────────────────────────
scene_names = sorted([
    d for d in os.listdir(SAMPLES_FOLDER)
    if os.path.isdir(os.path.join(SAMPLES_FOLDER, d))
])
print(f'\nFound {len(scene_names)} scenes: {scene_names}\n')

all_preds_global   = []
all_targets_global = []
scene_results      = {}

# ── Per-scene inference & evaluation ─────────────────────────────────────────
for loc in scene_names:
    print(f'{"="*60}')
    print(f'Scene: {loc}')
    print(f'{"="*60}')

    try:
        features, mask, bands = load_sample(loc)
    except Exception as e:
        print(f'  ERROR loading {loc}: {e}')
        continue

    print(f'  Image shape : {features.shape}   Mask shape: {mask.shape}')

    # Predict
    pred_img = sliding_window_predict(features.astype(np.float32))

    # NDWI water override (same as notebook)
    ndwi = (bands[2] - bands[7]) / (bands[2] + bands[7] + 1e-10)
    pred_img[ndwi > 0.20] = 3
    pred_img[mask == 0]   = 0    # restore unknown border

    # Thematic map
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(mask,     cmap=cmap, norm=norm); axes[0].set_title('Ground Truth');     axes[0].axis('off')
    im = axes[1].imshow(pred_img, cmap=cmap, norm=norm); axes[1].set_title('U-Net Prediction'); axes[1].axis('off')
    cbar = plt.colorbar(im, ax=axes, ticks=[0.5,1.5,2.5,3.5,4.5], fraction=0.02, pad=0.04)
    cbar.ax.set_yticklabels(['Unknown','Greenery','Sand','Water','Cement'])
    plt.suptitle(f'U-Net Land Cover Map — {loc}')
    plt.tight_layout()
    map_path = os.path.join(OUTPUT_FOLDER, f'{loc}_unet_prediction.png')
    plt.savefig(map_path, dpi=150)
    plt.close()
    print(f'  Thematic map saved → {map_path}')

    # Metrics (labeled pixels only)
    valid   = mask.flatten() > 0
    targets = mask.flatten()[valid]
    preds   = pred_img.flatten()[valid]

    all_preds_global.append(preds)
    all_targets_global.append(targets)

    iou_dict, miou = compute_miou(targets, preds)
    scene_results[loc] = {'miou': miou, 'iou': iou_dict}

    print(f'\n  mIoU: {miou:.4f}')
    for label, name in zip(CLASS_LABELS, CLASS_NAMES):
        print(f'    {name:<12}: IoU = {iou_dict[label]:.4f}')

    # Confusion matrix
    cm = confusion_matrix(targets, preds, labels=CLASS_LABELS)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, cmap='Greens')
    ax.set_title(f'U-Net Confusion Matrix — {loc}')
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_FOLDER, f'{loc}_confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f'  Confusion matrix saved → {cm_path}')

    print(f'\n  Errors of omission:')
    for i, name in enumerate(CLASS_NAMES):
        err = 1 - cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f'    {name}: {err:.4f}')

    print(f'\n  Errors of commission:')
    for i, name in enumerate(CLASS_NAMES):
        err = 1 - cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
        print(f'    {name}: {err:.4f}')

    print(f'\n{classification_report(targets, preds, target_names=CLASS_NAMES, labels=CLASS_LABELS)}')


# ── Overall summary ───────────────────────────────────────────────────────────
print('\n' + '='*65)
print('OVERALL SUMMARY (all samples_prepared scenes combined)')
print('='*65)

if all_preds_global:
    all_preds_cat   = np.concatenate(all_preds_global)
    all_targets_cat = np.concatenate(all_targets_global)

    iou_dict_all, miou_all = compute_miou(all_targets_cat, all_preds_cat)

    print(f'\n{"Scene":<20} {"mIoU":>8}', '  '.join(f'{n:>10}' for n in CLASS_NAMES))
    print('-'*80)
    for loc, res in scene_results.items():
        ious = '  '.join(f'{res["iou"][c]:>10.4f}' for c in CLASS_LABELS)
        print(f'{loc:<20} {res["miou"]:>8.4f}  {ious}')
    print('-'*80)
    print(f'{"OVERALL":<20} {miou_all:>8.4f}  ' +
          '  '.join(f'{iou_dict_all[c]:>10.4f}' for c in CLASS_LABELS))

    # Save summary table
    summary_path = os.path.join(OUTPUT_FOLDER, 'samples_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('U-Net evaluation on samples_prepared\n')
        f.write('='*80 + '\n')
        f.write(f'{"Scene":<20} {"mIoU":>8}  ' +
                '  '.join(f'{n:>10}' for n in CLASS_NAMES) + '\n')
        f.write('-'*80 + '\n')
        for loc, res in scene_results.items():
            ious = '  '.join(f'{res["iou"][c]:>10.4f}' for c in CLASS_LABELS)
            f.write(f'{loc:<20} {res["miou"]:>8.4f}  {ious}\n')
        f.write('-'*80 + '\n')
        f.write(f'{"OVERALL":<20} {miou_all:>8.4f}  ' +
                '  '.join(f'{iou_dict_all[c]:>10.4f}' for c in CLASS_LABELS) + '\n')
    print(f'\nSummary saved → {summary_path}')

print('\nDone.')
