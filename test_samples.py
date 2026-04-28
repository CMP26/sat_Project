"""
Test U-Net AND Minimum Distance Classifier (MDC) on all scenes
in the samples_prepared folder.

MDC:
  - Loads models/mdc_model.pkl + models/scaler.pkl (saved in the notebook).
  - Per-pixel classification using the same 17-feature vector.

U-Net:
  - Loads models/unet_best.pth (trained in the notebook).
  - Sliding-window inference with NDWI water override.

Outputs per scene  (models/samples_results/):
  - {loc}_comparison.png          — 3-panel: Ground Truth | MDC | U-Net
  - {loc}_mdc_confusion.png       — MDC confusion matrix
  - {loc}_unet_confusion.png      — U-Net confusion matrix

Global outputs:
  - samples_summary.txt           — mIoU table for both models
  - samples_comparison_bar.png    — per-class IoU bar chart
"""

import os
import numpy as np
import torch
import torch.nn as nn
import rasterio
import joblib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLES_FOLDER  = 'samples_prepared'
MODEL_FOLDER    = 'models'
BEST_MODEL_PATH = os.path.join(MODEL_FOLDER, 'unet_best.pth')
MDC_MODEL_PATH  = os.path.join(MODEL_FOLDER, 'mdc_model.pkl')
SCALER_PATH     = os.path.join(MODEL_FOLDER, 'scaler.pkl')
OUTPUT_FOLDER   = os.path.join(MODEL_FOLDER, 'samples_results')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

PATCH_SIZE   = 128
STRIDE       = 64
IN_CHANNELS  = 17
NUM_CLASSES  = 5
CLASS_NAMES  = ['Greenery', 'Sand', 'Water', 'Cement']
CLASS_LABELS = [1, 2, 3, 4]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

# ── Colour map (shared) ───────────────────────────────────────────────────────
COLORS = ['black', 'darkgreen', 'yellow', 'darkblue', 'gray']
CMAP   = mcolors.ListedColormap(COLORS)
NORM   = mcolors.BoundaryNorm([0, 1, 2, 3, 4, 5], CMAP.N)

# ═════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═════════════════════════════════════════════════════════════════════════════

def compute_indices(bands):
    B2  = bands[1].astype(float);  B3  = bands[2].astype(float)
    B4  = bands[3].astype(float);  B8  = bands[7].astype(float)
    B8A = bands[8].astype(float);  B11 = bands[10].astype(float)
    e   = 1e-10
    NDVI     = (B8  - B4)  / (B8  + B4  + e)
    MOISTURE = (B8A - B11) / (B8A + B11 + e)
    NDWI     = (B3  - B8)  / (B3  + B8  + e)
    NDBI     = (B11 - B8)  / (B11 + B8  + e)
    BSI      = ((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2) + e)
    return np.stack([NDVI, MOISTURE, NDWI, NDBI, BSI], axis=0)


def load_sample(location_name):
    """Returns features (17,H,W), mask (H,W), bands (12,H,W)."""
    spectral_path = os.path.join(SAMPLES_FOLDER, location_name,
                                 f'{location_name}_Spectral.tif')
    mask_path     = os.path.join(SAMPLES_FOLDER, location_name,
                                 f'{location_name}_Mask.tif')
    with rasterio.open(spectral_path) as f:
        bands = f.read().astype(np.float32)
    with rasterio.open(mask_path) as f:
        mask = f.read(1)
    H = min(bands.shape[1], mask.shape[0])
    W = min(bands.shape[2], mask.shape[1])
    bands = bands[:, :H, :W];  mask = mask[:H, :W]
    bands = np.clip(bands / 10000.0, 0, 1)
    bands = np.nan_to_num(bands, nan=0.0, posinf=0.0, neginf=0.0)
    indices  = np.nan_to_num(compute_indices(bands), nan=0.0, posinf=0.0, neginf=0.0)
    features = np.concatenate([bands, indices], axis=0)   # (17, H, W)
    return features, mask, bands


def compute_miou(true_labels, predicted_labels):
    iou_scores = {}
    for c in CLASS_LABELS:
        tp = ((predicted_labels == c) & (true_labels == c)).sum()
        fp = ((predicted_labels == c) & (true_labels != c)).sum()
        fn = ((predicted_labels != c) & (true_labels == c)).sum()
        iou_scores[c] = float(tp) / (tp + fp + fn + 1e-10)
    return iou_scores, float(np.mean(list(iou_scores.values())))


def save_confusion_matrix(cm, title, save_path, cmap_name):
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, cmap=cmap_name)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150);  plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# MDC  (load saved model + scaler from notebook)
# ═════════════════════════════════════════════════════════════════════════════

# Must be defined here so joblib can unpickle the saved object
class MinimumDistanceClassifier:
    def __init__(self):
        self.signatures = {}

    def train(self, X, y):
        for label in np.unique(y):
            self.signatures[label] = X[y == label].mean(axis=0)
        return self

    def classify(self, X):
        labels = np.array(list(self.signatures.keys()))
        means  = np.array(list(self.signatures.values()))
        batch_size  = 50000
        predictions = []
        for start in range(0, len(X), batch_size):
            batch       = X[start:start + batch_size]
            differences = batch[:, np.newaxis, :] - means[np.newaxis, :, :]
            distances   = np.sqrt((differences ** 2).sum(axis=2))
            closest     = np.argmin(distances, axis=1)
            predictions.append(labels[closest])
        return np.concatenate(predictions)


print(f'Loading MDC  from {MDC_MODEL_PATH}')
print(f'Loading scaler from {SCALER_PATH}')
mdc    = joblib.load(MDC_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


def mdc_predict_scene(features, mask):
    """Scale features, run MDC pixel-by-pixel, restore unknown border."""
    H, W = mask.shape
    X    = features.reshape(IN_CHANNELS, -1).T          # (H*W, 17)
    X_s  = scaler.transform(X)
    X_s  = np.nan_to_num(X_s, nan=0.0)
    pred = mdc.classify(X_s).reshape(H, W)
    pred[mask == 0] = 0
    return pred


# ═════════════════════════════════════════════════════════════════════════════
# U-Net  (load saved checkpoint from notebook)
# ═════════════════════════════════════════════════════════════════════════════

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


print(f'Loading U-Net from {BEST_MODEL_PATH}')
unet = UNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(DEVICE)
unet.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
unet.eval()


def unet_predict_scene(features, mask, bands):
    """Sliding-window U-Net prediction with NDWI water override."""
    _, H, W   = features.shape
    logit_sum = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    with torch.no_grad():
        for r in range(0, H - PATCH_SIZE + 1, STRIDE):
            for c in range(0, W - PATCH_SIZE + 1, STRIDE):
                patch = features[:, r:r+PATCH_SIZE, c:c+PATCH_SIZE]
                t     = torch.from_numpy(patch).unsqueeze(0).to(DEVICE)
                logit = unet(t).squeeze(0).cpu().numpy()
                logit_sum[:, r:r+PATCH_SIZE, c:c+PATCH_SIZE] += logit
                count_map[r:r+PATCH_SIZE, c:c+PATCH_SIZE]    += 1.0

    count_map = np.maximum(count_map, 1)
    pred      = (logit_sum / count_map[np.newaxis]).argmax(axis=0)

    ndwi = (bands[2] - bands[7]) / (bands[2] + bands[7] + 1e-10)
    pred[ndwi > 0.20] = 3
    pred[mask == 0]   = 0
    return pred


# ═════════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ═════════════════════════════════════════════════════════════════════════════

scene_names = sorted([
    d for d in os.listdir(SAMPLES_FOLDER)
    if os.path.isdir(os.path.join(SAMPLES_FOLDER, d))
])
print(f'\nFound {len(scene_names)} scenes: {scene_names}\n')

mdc_preds_all,  mdc_targets_all  = [], []
unet_preds_all, unet_targets_all = [], []
mdc_scene_results  = {}
unet_scene_results = {}

for loc in scene_names:
    print(f'\n{"="*65}')
    print(f'Scene: {loc}')
    print(f'{"="*65}')

    try:
        features, mask, bands = load_sample(loc)
    except Exception as e:
        print(f'  ERROR loading {loc}: {e}')
        continue

    print(f'  Shape: features={features.shape}  mask={mask.shape}')

    # ── Predict ──────────────────────────────────────────────────────────────
    mdc_pred  = mdc_predict_scene(features, mask)
    unet_pred = unet_predict_scene(features.astype(np.float32), mask, bands)

    # ── 3-panel comparison map ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes[0].imshow(mask,      cmap=CMAP, norm=NORM); axes[0].set_title('Ground Truth');   axes[0].axis('off')
    axes[1].imshow(mdc_pred,  cmap=CMAP, norm=NORM); axes[1].set_title('MDC Prediction'); axes[1].axis('off')
    im = axes[2].imshow(unet_pred, cmap=CMAP, norm=NORM); axes[2].set_title('U-Net Prediction'); axes[2].axis('off')
    cbar = plt.colorbar(im, ax=axes, ticks=[0.5,1.5,2.5,3.5,4.5], fraction=0.015, pad=0.03)
    cbar.ax.set_yticklabels(['Unknown', 'Greenery', 'Sand', 'Water', 'Cement'])
    plt.suptitle(f'Land Cover Map — {loc}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'{loc}_comparison.png'), dpi=150)
    plt.close()
    print(f'  Comparison map saved.')

    # ── Per-scene metrics for each model ─────────────────────────────────────
    valid   = mask.flatten() > 0
    targets = mask.flatten()[valid]

    for model_name, pred_img, results_dict, preds_acc, targets_acc, cm_color in [
        ('MDC',   mdc_pred,  mdc_scene_results,  mdc_preds_all,  mdc_targets_all,  'Blues'),
        ('U-Net', unet_pred, unet_scene_results, unet_preds_all, unet_targets_all, 'Greens'),
    ]:
        preds = pred_img.flatten()[valid]
        preds_acc.append(preds)
        targets_acc.append(targets)

        iou_dict, miou = compute_miou(targets, preds)
        results_dict[loc] = {'miou': miou, 'iou': iou_dict}

        cm = confusion_matrix(targets, preds, labels=CLASS_LABELS)
        save_confusion_matrix(
            cm,
            f'{model_name} Confusion Matrix — {loc}',
            os.path.join(OUTPUT_FOLDER,
                         f'{loc}_{model_name.lower().replace("-","")}_confusion.png'),
            cm_color,
        )

        print(f'\n  [{model_name}] mIoU: {miou:.4f}')
        for label, name in zip(CLASS_LABELS, CLASS_NAMES):
            print(f'    {name:<12}: IoU = {iou_dict[label]:.4f}')

        print(f'\n  [{model_name}] Errors of omission:')
        for i, name in enumerate(CLASS_NAMES):
            err = 1 - cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
            print(f'    {name}: {err:.4f}')

        print(f'\n  [{model_name}] Errors of commission:')
        for i, name in enumerate(CLASS_NAMES):
            err = 1 - cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
            print(f'    {name}: {err:.4f}')

        print(f'\n{classification_report(targets, preds, target_names=CLASS_NAMES, labels=CLASS_LABELS)}')


# ═════════════════════════════════════════════════════════════════════════════
# Global summary
# ═════════════════════════════════════════════════════════════════════════════

print('\n' + '='*75)
print('OVERALL SUMMARY — MDC vs U-Net across all samples_prepared scenes')
print('='*75)

mdc_iou_all,  mdc_miou_all  = compute_miou(np.concatenate(mdc_targets_all),
                                            np.concatenate(mdc_preds_all))
unet_iou_all, unet_miou_all = compute_miou(np.concatenate(unet_targets_all),
                                            np.concatenate(unet_preds_all))

# ── Console table ─────────────────────────────────────────────────────────────
print(f'\n{"Scene":<20} {"MDC mIoU":>10} {"UNet mIoU":>10}')
print('-' * 44)
for loc in scene_names:
    if loc not in mdc_scene_results:
        continue
    print(f'{loc:<20} {mdc_scene_results[loc]["miou"]:>10.4f} '
          f'{unet_scene_results[loc]["miou"]:>10.4f}')
print('-' * 44)
print(f'{"OVERALL":<20} {mdc_miou_all:>10.4f} {unet_miou_all:>10.4f}')

print(f'\nPer-class IoU (overall):')
print(f'  {"Class":<12} {"MDC":>8} {"U-Net":>8}')
print('  ' + '-' * 30)
for label, name in zip(CLASS_LABELS, CLASS_NAMES):
    print(f'  {name:<12} {mdc_iou_all[label]:>8.4f} {unet_iou_all[label]:>8.4f}')

# ── Bar chart ─────────────────────────────────────────────────────────────────
x     = np.arange(len(CLASS_NAMES))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width/2, [mdc_iou_all[c]  for c in CLASS_LABELS], width,
       label='MDC',   color='steelblue',  alpha=0.85)
ax.bar(x + width/2, [unet_iou_all[c] for c in CLASS_LABELS], width,
       label='U-Net', color='darkorange', alpha=0.85)
ax.axhline(mdc_miou_all,  color='steelblue',  linestyle='--', linewidth=1.2,
           label=f'MDC mIoU  {mdc_miou_all:.3f}')
ax.axhline(unet_miou_all, color='darkorange', linestyle='--', linewidth=1.2,
           label=f'U-Net mIoU {unet_miou_all:.3f}')
ax.set_xticks(x);  ax.set_xticklabels(CLASS_NAMES)
ax.set_ylabel('IoU');  ax.set_ylim(0, 1)
ax.set_title('Per-class IoU: MDC vs U-Net (samples_prepared)')
ax.legend();  plt.tight_layout()
bar_path = os.path.join(OUTPUT_FOLDER, 'samples_comparison_bar.png')
plt.savefig(bar_path, dpi=150);  plt.close()
print(f'\nBar chart saved → {bar_path}')

# ── Text summary file ─────────────────────────────────────────────────────────
summary_path = os.path.join(OUTPUT_FOLDER, 'samples_summary.txt')
with open(summary_path, 'w') as f:
    f.write('MDC vs U-Net evaluation on samples_prepared\n')
    f.write('=' * 65 + '\n')
    f.write(f'{"Metric":<35} {"MDC":>13} {"U-Net":>13}\n')
    f.write('=' * 65 + '\n')
    f.write(f'{"Overall mIoU":<35} {mdc_miou_all:>13.4f} {unet_miou_all:>13.4f}\n')
    for label, name in zip(CLASS_LABELS, CLASS_NAMES):
        f.write(f'  IoU — {name:<29} {mdc_iou_all[label]:>13.4f} {unet_iou_all[label]:>13.4f}\n')
    f.write('=' * 65 + '\n\n')
    f.write('Per-scene mIoU\n')
    f.write('-' * 45 + '\n')
    f.write(f'{"Scene":<20} {"MDC":>10} {"U-Net":>10}\n')
    f.write('-' * 45 + '\n')
    for loc in scene_names:
        if loc not in mdc_scene_results:
            continue
        f.write(f'{loc:<20} {mdc_scene_results[loc]["miou"]:>10.4f} '
                f'{unet_scene_results[loc]["miou"]:>10.4f}\n')
    f.write('-' * 45 + '\n')
print(f'Summary saved → {summary_path}')

print('\nDone.')
