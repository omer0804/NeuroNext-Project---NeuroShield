import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import random

# Set seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load data
bpm_df = pd.read_csv("data/processed/nightmares_model/data_for_model_bpm.csv")
motion_df = pd.read_csv("data/processed/nightmares_model/data_for_model_motion.csv")

# Feature extraction: last 120 seconds of each REM segment, variable length, padded

def build_rem_windows_features_padded(bpm_df, motion_df, rem_windows_csv, window_length=120):
    rem_windows = pd.read_csv(rem_windows_csv)
    features_list = []
    lengths_list = []
    labels_list = []
    groups_list = []

    for idx, row in rem_windows.iterrows():
        uid = row['id']
        rem_end = row['rem_end']
        rem_start = rem_end - window_length
        label = row['rem_type']

        bpm_sub = bpm_df[bpm_df['id'] == uid]
        motion_sub = motion_df[motion_df['id'] == uid]

        bpm_window = bpm_sub[(bpm_sub['time'] >= rem_start) & (bpm_sub['time'] <= rem_end)]
        motion_window = motion_sub[(motion_sub['time'] >= rem_start) & (motion_sub['time'] <= rem_end)]

        if len(bpm_window) == 0 or len(motion_window) == 0:
            continue

        sampling_points = min(len(bpm_window), len(motion_window))
        bpm_idx = np.linspace(0, len(bpm_window)-1, sampling_points).astype(int)
        motion_idx = np.linspace(0, len(motion_window)-1, sampling_points).astype(int)

        bpm_series = bpm_window['bpm'].values[bpm_idx]
        acc_xyz = motion_window[['acceleration_x', 'acceleration_y', 'acceleration_z']].values[motion_idx]

        mean_bpm = np.mean(bpm_series)
        std_bpm = np.std(bpm_series)
        max_bpm = np.max(bpm_series)
        min_bpm = np.min(bpm_series)
        gradient_bpm = np.gradient(bpm_series)
        mean_gradient_bpm = np.mean(gradient_bpm)
        magnitude = np.linalg.norm(acc_xyz, axis=1)
        magnitude_gradient = np.gradient(magnitude)
        energy_motion = np.sum(acc_xyz**2, axis=1)
        jerk_x = np.gradient(acc_xyz[:, 0])
        jerk_y = np.gradient(acc_xyz[:, 1])
        jerk_z = np.gradient(acc_xyz[:, 2])

        segment = []
        for i in range(sampling_points):
            feature_vec = [
                bpm_series[i],
                acc_xyz[i, 0], acc_xyz[i, 1], acc_xyz[i, 2],
                mean_bpm, std_bpm, max_bpm, min_bpm, mean_gradient_bpm,
                gradient_bpm[i], magnitude[i], magnitude_gradient[i],
                energy_motion[i], jerk_x[i], jerk_y[i], jerk_z[i]
            ]
            segment.append(feature_vec)

        segment_tensor = torch.tensor(segment, dtype=torch.float32)
        features_list.append(segment_tensor)
        lengths_list.append(segment_tensor.shape[0])
        labels_list.append(label)
        groups_list.append(uid)

    X_padded = pad_sequence(features_list, batch_first=True)  # (N, max_seq_len, F)
    y_tensor = torch.tensor(labels_list, dtype=torch.long)
    lengths_tensor = torch.tensor(lengths_list, dtype=torch.long)
    group_ids = np.array(groups_list)

    return X_padded, y_tensor, lengths_tensor, group_ids

# Load features
X, y, lengths, group_ids = build_rem_windows_features_padded(
    bpm_df, motion_df,
    rem_windows_csv="data/processed/nightmares_model/rem_windows_and_classification.csv",
    window_length=120
)

# Oversample minority class (label=1)
def oversample_minority_class_tensor(X, y, lengths, group_ids, factor=2):
    minority_indices = (y == 1).nonzero(as_tuple=True)[0]
    oversampled_indices = minority_indices.repeat(factor - 1)
    X_aug = torch.cat([X, X[oversampled_indices]], dim=0)
    y_aug = torch.cat([y, y[oversampled_indices]], dim=0)
    lengths_aug = torch.cat([lengths, lengths[oversampled_indices]], dim=0)
    group_ids_aug = np.concatenate([group_ids, group_ids[oversampled_indices.cpu().numpy()]], axis=0)
    return X_aug, y_aug, lengths_aug, group_ids_aug

X, y, lengths, group_ids = oversample_minority_class_tensor(X, y, lengths, group_ids, factor=2)

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y.numpy())

weights = torch.tensor(class_weights, dtype=torch.float32)

# Conv1D model with masking
class Conv1DClassifier(nn.Module):
    def __init__(self, input_channels, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x, lengths):
        mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]
        mask = mask.float().unsqueeze(1)  # (batch, 1, time)

        x = x.transpose(1, 2)  # (batch, features, time)
        x = self.relu1(self.conv1(x)) * mask
        x = self.relu2(self.conv2(x)) * mask

        # masked average pooling
        masked_sum = torch.sum(x * mask, dim=2)
        masked_count = torch.sum(mask, dim=2) + 1e-6
        x = masked_sum / masked_count

        return self.fc(x)

# Training and evaluation
logo = LeaveOneGroupOut()
all_preds, all_probs, all_targets, all_ids = [], [], [], []

for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=group_ids)):
    print(f"\nFold {fold + 1}: Test subject {group_ids[test_idx[0]]}")

    model = Conv1DClassifier(input_channels=X.shape[2])
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loader = DataLoader(TensorDataset(X[train_idx], y[train_idx], lengths[train_idx]), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X[test_idx], y[test_idx], lengths[test_idx]), batch_size=64)

    model.train()
    for epoch in range(5):
        for xb, yb, lb in train_loader:
            optimizer.zero_grad()
            out = model(xb, lb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        for xb, yb, lb in test_loader:
            logits = model(xb, lb)
            probs = softmax(logits, dim=1)[:, 1]
            preds = (probs > 0.5).int()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(yb.cpu().numpy())
            all_ids.extend([group_ids[test_idx[0]]] * len(yb))

# Evaluation metrics
all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

correct = (all_preds == all_targets).sum()
total = len(all_targets)
acc = correct / total

print("\n==============================")
print(f"Overall Accuracy: {acc:.2%} ({correct} out of {total})")
print("==============================")

false_negatives = ((all_preds == 0) & (all_targets == 1)).sum()
false_positives = ((all_preds == 1) & (all_targets == 0)).sum()
total_actual_1 = (all_targets == 1).sum()
total_actual_0 = (all_targets == 0).sum()

fn_rate = 100 * false_negatives / total_actual_1 if total_actual_1 else 0
fp_rate = 100 * false_positives / total_actual_0 if total_actual_0 else 0

print(f"\n🔴 False Negatives: {false_negatives} out of {total_actual_1} → {fn_rate:.2f}%")
print(f"🔵 False Positives: {false_positives} out of {total_actual_0} → {fp_rate:.2f}%")

precision = precision_score(all_targets, all_preds)
recall = recall_score(all_targets, all_preds)
f1 = f1_score(all_targets, all_preds)

print(f"\nPrecision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1 Score:  {f1:.2f}")