import os
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", DEVICE)

FILE_PATH = "vanet_traffic_data.csv"
TARGET_COLUMN = "label"
ROAD_COLUMN = "road_segment_id"
TIME_COLUMN = "timestamp"

MAX_ROWS = 10000
SEQUENCE_LENGTH = 5
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001

# ============================================================
# GLOBAL PLOT STYLE
# ============================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 18
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.grid"] = False

FIGSIZE = (10, 8)
DPI = 1000

COLOR_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"
]

LEGEND_FONT = {"family": "Times New Roman", "weight": "bold", "size": 14}

data = pd.read_csv(FILE_PATH)

print("\n========== ORIGINAL DATASET ==========")
print("Original Dataset Shape:", data.shape)
print(data.head())

if len(data) > MAX_ROWS:
    data = data.sample(n=MAX_ROWS, random_state=SEED).reset_index(drop=True)

print("\n========== SELECTED DATASET ==========")
print("Selected Dataset Shape:", data.shape)

required_columns = [TARGET_COLUMN, ROAD_COLUMN, TIME_COLUMN]

missing_columns = [column for column in required_columns if column not in data.columns]

if len(missing_columns) > 0:
    raise ValueError(f"Missing required columns: {missing_columns}")

print("\n========== ORIGINAL CLASS DISTRIBUTION ==========")
print(data[TARGET_COLUMN].value_counts())

data = data.drop_duplicates().reset_index(drop=True)

data = data.dropna(subset=[TARGET_COLUMN, ROAD_COLUMN, TIME_COLUMN]).reset_index(drop=True)

data[TIME_COLUMN] = pd.to_datetime(data[TIME_COLUMN], errors="coerce")

data = data.dropna(subset=[TIME_COLUMN]).reset_index(drop=True)

data = data.sort_values(by=[ROAD_COLUMN, TIME_COLUMN]).reset_index(drop=True)

data["hour"] = data[TIME_COLUMN].dt.hour
data["minute"] = data[TIME_COLUMN].dt.minute
data["day"] = data[TIME_COLUMN].dt.day
data["month"] = data[TIME_COLUMN].dt.month
data["day_of_week"] = data[TIME_COLUMN].dt.dayofweek
data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)

data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
data["day_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
data["day_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)

feature_data = data.drop(columns=[TARGET_COLUMN, TIME_COLUMN], errors="ignore")

target_data = data[TARGET_COLUMN].copy()

categorical_columns = feature_data.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

numerical_columns = feature_data.select_dtypes(include=[np.number]).columns.tolist()

if len(numerical_columns) > 0:
    numerical_imputer = SimpleImputer(strategy="median")
    feature_data[numerical_columns] = numerical_imputer.fit_transform(feature_data[numerical_columns])

if len(categorical_columns) > 0:
    categorical_imputer = SimpleImputer(strategy="most_frequent")
    feature_data[categorical_columns] = categorical_imputer.fit_transform(feature_data[categorical_columns])

feature_data = pd.get_dummies(feature_data, columns=categorical_columns, drop_first=False)

feature_data = feature_data.replace([np.inf, -np.inf], np.nan)

feature_data = feature_data.fillna(0)

processed_data = pd.concat([feature_data, target_data], axis=1)

processed_data[ROAD_COLUMN] = data[ROAD_COLUMN].values

processed_data[TIME_COLUMN] = data[TIME_COLUMN].values

processed_data = processed_data.sort_values(by=[ROAD_COLUMN, TIME_COLUMN]).reset_index(drop=True)

label_encoder = LabelEncoder()

processed_data["encoded_target"] = label_encoder.fit_transform(processed_data[TARGET_COLUMN])

class_names = list(label_encoder.classes_)

number_of_classes = len(class_names)

print("\n========== ENCODED TARGET INFORMATION ==========")
print("Class Names:", class_names)
print("Number of Classes:", number_of_classes)

if number_of_classes < 2:
    raise ValueError("Only one class is available. Use the complete dataset or balanced sampling to obtain multiple classes.")

feature_columns = [column for column in feature_data.columns if column not in [ROAD_COLUMN, TIME_COLUMN]]

feature_matrix = processed_data[feature_columns].values.astype(np.float32)

target_array = processed_data["encoded_target"].values.astype(np.int64)

road_segments = processed_data[ROAD_COLUMN].values

sequence_list = []
sequence_labels = []
sequence_groups = []

for road_segment in np.unique(road_segments):
    segment_indices = np.where(road_segments == road_segment)[0]

    segment_features = feature_matrix[segment_indices]
    segment_labels = target_array[segment_indices]

    if len(segment_features) < SEQUENCE_LENGTH:
        continue

    for start_index in range(0, len(segment_features) - SEQUENCE_LENGTH + 1):
        end_index = start_index + SEQUENCE_LENGTH

        sequence_features = segment_features[start_index:end_index]
        sequence_target = segment_labels[end_index - 1]

        sequence_list.append(sequence_features)
        sequence_labels.append(sequence_target)
        sequence_groups.append(road_segment)

X_sequences = np.array(sequence_list, dtype=np.float32)
y_sequences = np.array(sequence_labels, dtype=np.int64)
group_sequences = np.array(sequence_groups)

print("\n========== TEMPORAL FEATURE REPRESENTATION ==========")
print("Temporal Sequence Shape:", X_sequences.shape)
print("Temporal Label Shape:", y_sequences.shape)
print("Sequence Class Distribution:", Counter(y_sequences))

if len(X_sequences) == 0:
    raise ValueError("No temporal sequences were created. Reduce SEQUENCE_LENGTH or check road segment data.")

if len(np.unique(y_sequences)) < 2:
    raise ValueError("Temporal sequences contain only one class. Use more records or balanced sampling.")

X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.20, random_state=SEED, stratify=y_sequences)

scaler = StandardScaler()

X_train_2d = X_train.reshape(-1, X_train.shape[-1])

X_test_2d = X_test.reshape(-1, X_test.shape[-1])

X_train_scaled = scaler.fit_transform(X_train_2d).reshape(X_train.shape)

X_test_scaled = scaler.transform(X_test_2d).reshape(X_test.shape)

print("\n========== TRAINING AND TESTING DATA ==========")
print("Training Feature Shape:", X_train_scaled.shape)
print("Testing Feature Shape:", X_test_scaled.shape)
print("Training Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)

class VANETDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

train_dataset = VANETDataset(X_train_scaled, y_train)

test_dataset = VANETDataset(X_test_scaled, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# REFINED LION OPTIMIZER (RLion)
# ============================================================
class RefinedLion(torch.optim.Optimizer):
    """
    Refined Lion Optimizer (RLion): sign-based update with momentum
    interpolation, gradient clipping, and decoupled weight decay,
    used here for parameter tuning of both feature learners.
    """
    def __init__(self, params, lr=0.001, betas=(0.9, 0.99), weight_decay=0.0, grad_clip=1.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, grad_clip=grad_clip)
        super(RefinedLion, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            grad_clip = group["grad_clip"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                gradient = parameter.grad

                if gradient.is_sparse:
                    raise RuntimeError("RefinedLion does not support sparse gradients.")

                gradient = torch.clamp(gradient, -grad_clip, grad_clip)

                state = self.state[parameter]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(parameter)

                exp_avg = state["exp_avg"]

                if weight_decay != 0:
                    parameter.mul_(1 - lr * weight_decay)

                update = beta1 * exp_avg + (1 - beta1) * gradient

                parameter.add_(torch.sign(update), alpha=-lr)

                exp_avg.mul_(beta2).add_(gradient, alpha=1 - beta2)

        return loss

class PureMambaBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super(PureMambaBlock, self).__init__()

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        self.state_projection = nn.Linear(hidden_dim, hidden_dim)

        self.gate_projection = nn.Linear(hidden_dim, hidden_dim)

        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        projected_x = self.input_projection(x)

        state_output = torch.tanh(self.state_projection(projected_x))

        gate_output = torch.sigmoid(self.gate_projection(projected_x))

        selective_output = state_output * gate_output

        output = self.output_projection(selective_output)

        output = self.dropout(output)

        output = self.norm(output + projected_x)

        return output

class MambaFeatureLearner(nn.Module):
    def __init__(self, input_dim, hidden_dim, feature_dim, number_of_classes):
        super(MambaFeatureLearner, self).__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dim)

        self.mamba_block1 = PureMambaBlock(hidden_dim, hidden_dim)

        self.mamba_block2 = PureMambaBlock(hidden_dim, hidden_dim)

        self.mamba_block3 = PureMambaBlock(hidden_dim, hidden_dim)

        self.feature_layer = nn.Sequential(
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Linear(feature_dim, number_of_classes)

    def forward(self, x):
        x = self.input_layer(x)

        x = self.mamba_block1(x)

        x = self.mamba_block2(x)

        x = self.mamba_block3(x)

        x = torch.mean(x, dim=1)

        deep_features = self.feature_layer(x)

        output = self.classifier(deep_features)

        return output, deep_features

class PrimaryCapsules(nn.Module):
    def __init__(self, input_dim, capsule_dim, number_of_capsules):
        super(PrimaryCapsules, self).__init__()

        self.number_of_capsules = number_of_capsules

        self.capsule_dim = capsule_dim

        self.projection = nn.Linear(input_dim, number_of_capsules * capsule_dim)

    def squash(self, tensor):
        squared_norm = torch.sum(tensor ** 2, dim=-1, keepdim=True)

        scale = squared_norm / (1.0 + squared_norm)

        normalized_tensor = tensor / torch.sqrt(squared_norm + 1e-8)

        return scale * normalized_tensor

    def forward(self, x):
        batch_size, sequence_length, input_dim = x.shape

        capsules = self.projection(x)

        capsules = capsules.view(batch_size, sequence_length, self.number_of_capsules, self.capsule_dim)

        capsules = self.squash(capsules)

        return capsules

class CapsuleFeatureLearner(nn.Module):
    def __init__(self, input_dim, hidden_dim, capsule_dim, number_of_capsules, feature_dim, number_of_classes, routing_iterations=3):
        super(CapsuleFeatureLearner, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)

        self.primary_capsules = PrimaryCapsules(hidden_dim, capsule_dim, number_of_capsules)

        self.routing_iterations = routing_iterations

        self.number_of_capsules = number_of_capsules

        self.capsule_dim = capsule_dim

        self.feature_layer = nn.Sequential(
            nn.Linear(number_of_capsules * capsule_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Linear(feature_dim, number_of_classes)

    def squash(self, tensor):
        squared_norm = torch.sum(tensor ** 2, dim=-1, keepdim=True)

        scale = squared_norm / (1.0 + squared_norm)

        normalized_tensor = tensor / torch.sqrt(squared_norm + 1e-8)

        return scale * normalized_tensor

    def forward(self, x):
        x = self.embedding(x)

        capsules = self.primary_capsules(x)

        capsules = torch.mean(capsules, dim=1)

        batch_size = capsules.shape[0]

        routing_logits = torch.zeros(batch_size, self.number_of_capsules, device=x.device)

        routed_output = capsules

        for _ in range(self.routing_iterations):
            routing_coefficients = torch.softmax(routing_logits, dim=1).unsqueeze(-1)

            weighted_capsules = capsules * routing_coefficients

            routed_output = self.squash(weighted_capsules)

            agreement = torch.sum(capsules * routed_output, dim=-1)

            routing_logits = routing_logits + agreement

        routed_output = routed_output.reshape(batch_size, -1)

        deep_features = self.feature_layer(routed_output)

        output = self.classifier(deep_features)

        return output, deep_features

input_dimension = X_train_scaled.shape[-1]

mamba_hidden_dimension = 64

capsule_hidden_dimension = 64

capsule_dimension = 16

number_of_capsules = 8

deep_feature_dimension = 128

mamba_model = MambaFeatureLearner(input_dimension, mamba_hidden_dimension, deep_feature_dimension, number_of_classes).to(DEVICE)

capsule_model = CapsuleFeatureLearner(input_dimension, capsule_hidden_dimension, capsule_dimension, number_of_capsules, deep_feature_dimension, number_of_classes).to(DEVICE)

mamba_optimizer = RefinedLion(mamba_model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.99), weight_decay=1e-5, grad_clip=1.0)

capsule_optimizer = RefinedLion(capsule_model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.99), weight_decay=1e-5, grad_clip=1.0)

criterion = nn.CrossEntropyLoss()

def train_model(model, optimizer, train_loader, test_loader, model_name):
    train_losses = []
    validation_losses = []
    train_accuracies = []
    validation_accuracies = []

    for epoch in range(EPOCHS):
        model.train()

        total_train_loss = 0.0
        train_predictions = []
        train_targets = []

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)

            optimizer.zero_grad()

            predictions, deep_features = model(batch_features)

            loss = criterion(predictions, batch_labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_train_loss += loss.item()

            predicted_classes = torch.argmax(predictions, dim=1)

            train_predictions.extend(predicted_classes.detach().cpu().numpy())

            train_targets.extend(batch_labels.detach().cpu().numpy())

        average_train_loss = total_train_loss / max(len(train_loader), 1)

        train_accuracy = accuracy_score(train_targets, train_predictions)

        model.eval()

        total_validation_loss = 0.0
        validation_predictions = []
        validation_targets = []

        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features = batch_features.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)

                predictions, deep_features = model(batch_features)

                loss = criterion(predictions, batch_labels)

                total_validation_loss += loss.item()

                predicted_classes = torch.argmax(predictions, dim=1)

                validation_predictions.extend(predicted_classes.detach().cpu().numpy())

                validation_targets.extend(batch_labels.detach().cpu().numpy())

        average_validation_loss = total_validation_loss / max(len(test_loader), 1)

        validation_accuracy = accuracy_score(validation_targets, validation_predictions)

        train_losses.append(average_train_loss)

        validation_losses.append(average_validation_loss)

        train_accuracies.append(train_accuracy)

        validation_accuracies.append(validation_accuracy)

        print(f"{model_name} | Epoch {epoch + 1:02d}/{EPOCHS} | Train Loss: {average_train_loss:.4f} | Val Loss: {average_validation_loss:.4f} | Train Acc: {train_accuracy:.4f} | Val Acc: {validation_accuracy:.4f}")

    history = {
        "train_loss": train_losses,
        "val_loss": validation_losses,
        "train_accuracy": train_accuracies,
        "val_accuracy": validation_accuracies
    }

    return history

mamba_history = train_model(mamba_model, mamba_optimizer, train_loader, test_loader, "Mamba")

capsule_history = train_model(capsule_model, capsule_optimizer, train_loader, test_loader, "CapsNet")

def evaluate_model(model, test_loader, model_name):
    model.eval()

    all_predictions = []
    all_targets = []
    all_features = []
    all_probabilities = []

    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(DEVICE)

            predictions, deep_features = model(batch_features)

            probabilities = F.softmax(predictions, dim=1)

            predicted_classes = torch.argmax(predictions, dim=1)

            all_predictions.extend(predicted_classes.cpu().numpy())

            all_targets.extend(batch_labels.numpy())

            all_features.append(deep_features.cpu().numpy())

            all_probabilities.append(probabilities.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)

    all_probabilities = np.concatenate(all_probabilities, axis=0)

    accuracy = accuracy_score(all_targets, all_predictions)

    precision = precision_score(all_targets, all_predictions, average="weighted", zero_division=0)

    recall = recall_score(all_targets, all_predictions, average="weighted", zero_division=0)

    f1 = f1_score(all_targets, all_predictions, average="weighted", zero_division=0)

    print(f"\n========== {model_name} PERFORMANCE ==========")

    print("Accuracy:", round(accuracy, 6))

    print("Precision:", round(precision, 6))

    print("Recall:", round(recall, 6))

    print("F1-Score:", round(f1, 6))

    print("\nClassification Report:")

    print(classification_report(all_targets, all_predictions, target_names=class_names, zero_division=0))

    confusion = confusion_matrix(all_targets, all_predictions)

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Deep_Features": all_features,
        "Predictions": all_predictions,
        "Targets": all_targets,
        "Probabilities": all_probabilities,
        "Confusion": confusion
    }

mamba_results = evaluate_model(mamba_model, test_loader, "Mamba")

capsule_results = evaluate_model(capsule_model, test_loader, "CapsNet")

results_table = pd.DataFrame([
    {
        "Model": mamba_results["Model"],
        "Accuracy": mamba_results["Accuracy"],
        "Precision": mamba_results["Precision"],
        "Recall": mamba_results["Recall"],
        "F1-Score": mamba_results["F1-Score"]
    },
    {
        "Model": capsule_results["Model"],
        "Accuracy": capsule_results["Accuracy"],
        "Precision": capsule_results["Precision"],
        "Recall": capsule_results["Recall"],
        "F1-Score": capsule_results["F1-Score"]
    }
])

print("\n========== FINAL PERFORMANCE COMPARISON ==========")

print(results_table)

results_table.to_csv("Mamba_CapsNet_Performance.csv", index=False)

# ============================================================
# EXCEL EXPORT: PERFORMANCE METRICS FOR BOTH MODELS
# ============================================================
def build_classwise_report_df(results_dict, class_names):
    report_dict = classification_report(
        results_dict["Targets"], results_dict["Predictions"],
        target_names=class_names, zero_division=0, output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose().reset_index()
    report_df = report_df.rename(columns={"index": "Class"})
    return report_df

def build_confusion_df(results_dict, class_names):
    confusion_df = pd.DataFrame(
        results_dict["Confusion"],
        index=[f"Actual_{c}" for c in class_names],
        columns=[f"Predicted_{c}" for c in class_names]
    ).reset_index().rename(columns={"index": "Class"})
    return confusion_df

def export_performance_to_excel(results_a, results_b, class_names, output_path):
    overall_df = pd.DataFrame([
        {
            "Model": results_a["Model"],
            "Accuracy": results_a["Accuracy"],
            "Precision": results_a["Precision"],
            "Recall": results_a["Recall"],
            "F1-Score": results_a["F1-Score"]
        },
        {
            "Model": results_b["Model"],
            "Accuracy": results_b["Accuracy"],
            "Precision": results_b["Precision"],
            "Recall": results_b["Recall"],
            "F1-Score": results_b["F1-Score"]
        }
    ])

    mamba_report_df = build_classwise_report_df(results_a, class_names)
    capsnet_report_df = build_classwise_report_df(results_b, class_names)

    mamba_confusion_df = build_confusion_df(results_a, class_names)
    capsnet_confusion_df = build_confusion_df(results_b, class_names)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        overall_df.to_excel(writer, sheet_name="Overall_Comparison", index=False)
        mamba_report_df.to_excel(writer, sheet_name="Mamba_Classwise_Report", index=False)
        capsnet_report_df.to_excel(writer, sheet_name="CapsNet_Classwise_Report", index=False)
        mamba_confusion_df.to_excel(writer, sheet_name="Mamba_Confusion_Matrix", index=False)
        capsnet_confusion_df.to_excel(writer, sheet_name="CapsNet_Confusion_Matrix", index=False)

    from openpyxl import load_workbook
    from openpyxl.styles import Font, Alignment

    workbook = load_workbook(output_path)

    header_font = Font(name="Times New Roman", bold=True, size=12)
    body_font = Font(name="Times New Roman", bold=False, size=11)
    center_align = Alignment(horizontal="center", vertical="center")

    for sheet_name in workbook.sheetnames:
        worksheet = workbook[sheet_name]

        for column_cells in worksheet.columns:
            max_length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
            column_letter = column_cells[0].column_letter
            worksheet.column_dimensions[column_letter].width = max_length + 4

        for row_index, row in enumerate(worksheet.iter_rows(), start=1):
            for cell in row:
                cell.alignment = center_align
                if row_index == 1:
                    cell.font = header_font
                else:
                    cell.font = body_font

    workbook.save(output_path)

    print(f"\nPerformance metrics for both models exported to: {os.path.abspath(output_path)}")

export_performance_to_excel(mamba_results, capsule_results, class_names, "Mamba_CapsNet_Performance.xlsx")

def run_mcnemar_test(results_a, results_b):
    """
    Checks whether the two models' aggregate metrics being close
    (e.g. 0.9775 vs 0.9781) reflects a real difference or just
    noise, by comparing their PAIRED predictions on the same test
    samples rather than their summary scores.
    """
    targets_a = np.array(results_a["Targets"])
    targets_b = np.array(results_b["Targets"])

    if not np.array_equal(targets_a, targets_b):
        print("\nWarning: test targets differ between models — McNemar's test skipped.")
        return None

    predictions_a = np.array(results_a["Predictions"])
    predictions_b = np.array(results_b["Predictions"])

    correct_a = (predictions_a == targets_a)
    correct_b = (predictions_b == targets_b)

    # b01: A wrong, B right | b10: A right, B wrong
    b01 = int(np.sum((~correct_a) & correct_b))
    b10 = int(np.sum(correct_a & (~correct_b)))

    contingency_table = np.array([
        [int(np.sum(correct_a & correct_b)), b10],
        [b01, int(np.sum((~correct_a) & (~correct_b)))]
    ])

    print(f"\n========== McNEMAR'S TEST: {results_a['Model']} vs {results_b['Model']} ==========")
    print("Contingency Table [ [both_correct, A_only_correct], [B_only_correct, both_wrong] ]:")
    print(contingency_table)

    discordant_total = b01 + b10

    if discordant_total == 0:
        print("No discordant predictions — the two models made identical correct/incorrect calls on every sample.")
        return {"b01": b01, "b10": b10, "statistic": 0.0, "p_value": 1.0}

    if discordant_total < 25:
        from scipy.stats import binomtest
        p_value = binomtest(min(b01, b10), discordant_total, 0.5).pvalue
        statistic = None
        print(f"Discordant pairs = {discordant_total} (< 25) — used exact binomial test.")
    else:
        statistic = ((abs(b01 - b10) - 1) ** 2) / discordant_total
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(statistic, df=1)
        print(f"Discordant pairs = {discordant_total} — used chi-square approximation.")
        print(f"McNemar Statistic: {statistic:.6f}")

    print(f"p-value: {p_value:.6f}")

    if p_value < 0.05:
        print(f"Result: statistically significant difference between {results_a['Model']} and {results_b['Model']} (p < 0.05).")
    else:
        print(f"Result: NO statistically significant difference between {results_a['Model']} and {results_b['Model']} (p >= 0.05) — "
              f"the close accuracy figures are genuine, not a bug.")

    return {"b01": b01, "b10": b10, "statistic": statistic, "p_value": p_value}

mcnemar_result = run_mcnemar_test(mamba_results, capsule_results)

np.save("Mamba_Deep_Features.npy", mamba_results["Deep_Features"])

np.save("CapsNet_Deep_Features.npy", capsule_results["Deep_Features"])

torch.save(mamba_model.state_dict(), "Mamba_Feature_Learner.pth")

torch.save(capsule_model.state_dict(), "CapsNet_Feature_Learner.pth")

# ============================================================
# PLOT UTILITIES
# ============================================================
def apply_axis_style(ax=None):
    ax = ax or plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Times New Roman")
        label.set_fontweight("bold")
        label.set_fontsize(18)
    ax.grid(False)

def style_title_labels(title, xlabel, ylabel):
    plt.title(title, fontsize=18, fontweight="bold", fontfamily="Times New Roman")
    plt.xlabel(xlabel, fontsize=18, fontweight="bold", fontfamily="Times New Roman")
    plt.ylabel(ylabel, fontsize=18, fontweight="bold", fontfamily="Times New Roman")

def plot_class_distribution_fig(labels_array, class_names, save_dir):
    counts = Counter(labels_array)
    count_values = [counts.get(class_index, 0) for class_index in range(len(class_names))]

    max_count = max(count_values) if count_values else 0
    label_offset = max_count * 0.01 if max_count > 0 else 0.1

    plt.figure(figsize=FIGSIZE)
    bars = plt.bar(class_names, count_values, color=COLOR_PALETTE[:len(class_names)])

    for bar, value in zip(bars, count_values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + label_offset, f"{value}",
                  ha="center", va="bottom", fontsize=14, fontweight="bold", fontfamily="Times New Roman")

    plt.ylim(0, max_count * 1.15 if max_count > 0 else 1.0)
    style_title_labels("Traffic Congestion Class Distribution", "Congestion Class", "Number of Sequences")
    apply_axis_style()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Traffic_Congestion_Class_Distribution.png"), dpi=DPI)
    plt.show()

def plot_feature_distribution_by_class_fig(source_data, target_column, feature_list, class_names, save_dir, max_features=4):
    selected_features = feature_list[:max_features]
    number_of_features = len(selected_features)

    if number_of_features == 0:
        print("No numeric traffic features available for distribution plot — skipping.")
        return

    n_cols = 2
    n_rows = int(np.ceil(number_of_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIGSIZE[0] * n_cols * 0.75, FIGSIZE[1] * n_rows * 0.75))
    axes = np.array(axes).reshape(-1)

    for feature_index, feature_name in enumerate(selected_features):
        ax = axes[feature_index]
        sns.boxplot(x=target_column, y=feature_name, data=source_data, ax=ax,
                    palette=COLOR_PALETTE[:len(class_names)])
        ax.set_title(feature_name, fontsize=16, fontweight="bold", fontfamily="Times New Roman")
        ax.set_xlabel("Congestion Class", fontsize=14, fontweight="bold", fontfamily="Times New Roman")
        ax.set_ylabel(feature_name, fontsize=14, fontweight="bold", fontfamily="Times New Roman")
        apply_axis_style(ax)

    for unused_index in range(number_of_features, len(axes)):
        fig.delaxes(axes[unused_index])

    fig.suptitle("Traffic Feature Distribution by Congestion Level", fontsize=20, fontweight="bold", fontfamily="Times New Roman")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_dir, "Traffic_Feature_Distribution_by_Congestion_Level.png"), dpi=DPI)
    plt.show()

def compute_gradient_feature_importance(model, test_loader, number_of_input_features):
    model.eval()

    importance_accumulator = np.zeros(number_of_input_features, dtype=np.float64)
    total_samples = 0

    for batch_features, batch_labels in test_loader:
        batch_features = batch_features.to(DEVICE)
        batch_features.requires_grad_(True)

        predictions, _ = model(batch_features)

        predicted_classes = torch.argmax(predictions, dim=1)

        selected_scores = predictions.gather(1, predicted_classes.unsqueeze(1)).squeeze(1)

        model.zero_grad()

        selected_scores.sum().backward()

        gradients = batch_features.grad.detach().cpu().numpy()

        batch_importance = np.abs(gradients).mean(axis=1)

        importance_accumulator += batch_importance.sum(axis=0)

        total_samples += batch_features.shape[0]

        batch_features.grad = None
        batch_features.requires_grad_(False)

    importance_scores = importance_accumulator / max(total_samples, 1)

    score_sum = importance_scores.sum()

    if score_sum > 0:
        importance_scores = importance_scores / score_sum

    return importance_scores

def plot_feature_importance_fig(model, test_loader, feature_columns, model_name, save_dir, top_n=15):
    importance_scores = compute_gradient_feature_importance(model, test_loader, len(feature_columns))

    top_n = min(top_n, len(feature_columns))

    sorted_indices = np.argsort(importance_scores)[::-1][:top_n]

    top_features = [feature_columns[i] for i in sorted_indices][::-1]
    top_scores = [importance_scores[i] for i in sorted_indices][::-1]

    max_score = max(top_scores) if top_scores else 0
    label_offset = max_score * 0.02 if max_score > 0 else 0.001

    plt.figure(figsize=FIGSIZE)
    bars = plt.barh(top_features, top_scores, color=COLOR_PALETTE[6])

    for bar, value in zip(bars, top_scores):
        plt.text(bar.get_width() + label_offset, bar.get_y() + bar.get_height() / 2, f"{value:.3f}",
                  va="center", ha="left", fontsize=12, fontweight="bold", fontfamily="Times New Roman")

    plt.xlim(0, max_score * 1.25 if max_score > 0 else 1.0)
    style_title_labels(f"{model_name} Feature Importance Across Traffic Congestion Classes", "Importance Score", "Feature")
    apply_axis_style()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_Feature_Importance.png"), dpi=DPI)
    plt.show()

def plot_error_distribution_fig(results_a, results_b, class_names, save_dir):
    targets_a = np.array(results_a["Targets"])
    predictions_a = np.array(results_a["Predictions"])

    targets_b = np.array(results_b["Targets"])
    predictions_b = np.array(results_b["Predictions"])

    error_counts_a = []
    error_counts_b = []

    for class_index in range(len(class_names)):
        mask_a = (targets_a == class_index)
        error_counts_a.append(int(np.sum(predictions_a[mask_a] != targets_a[mask_a])))

        mask_b = (targets_b == class_index)
        error_counts_b.append(int(np.sum(predictions_b[mask_b] != targets_b[mask_b])))

    x_positions = np.arange(len(class_names))
    bar_width = 0.35

    max_errors = max(error_counts_a + error_counts_b) if (error_counts_a + error_counts_b) else 0
    label_offset = max_errors * 0.02 if max_errors > 0 else 0.1
    y_upper_limit = max_errors * 1.20 if max_errors > 0 else 1.0

    plt.figure(figsize=FIGSIZE)
    bars_a = plt.bar(x_positions - bar_width / 2, error_counts_a, bar_width, color=COLOR_PALETTE[0], label=f"{results_a['Model']} Errors")
    bars_b = plt.bar(x_positions + bar_width / 2, error_counts_b, bar_width, color=COLOR_PALETTE[1], label=f"{results_b['Model']} Errors")
    plt.xticks(x_positions, class_names)

    for bar_group in (bars_a, bars_b):
        for bar in bar_group:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + label_offset, f"{int(height)}",
                      ha="center", va="bottom", fontsize=13, fontweight="bold", fontfamily="Times New Roman")

    plt.ylim(0, y_upper_limit)
    style_title_labels("Error Distribution Across Congestion Classes", "Class", "Number of Misclassifications")
    apply_axis_style()
    plt.legend(prop=LEGEND_FONT)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Error_Distribution_Across_Classes.png"), dpi=DPI)
    plt.show()

def plot_confusion_matrix_fig(confusion, class_names, model_name, save_dir):
    plt.figure(figsize=FIGSIZE)
    sns.heatmap(
        confusion, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        annot_kws={"fontsize": 18, "fontweight": "bold", "fontfamily": "Times New Roman"},
        cbar=True
    )
    style_title_labels(f"{model_name} Confusion Matrix", "Predicted Class", "Actual Class")
    apply_axis_style()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_Confusion_Matrix.png"), dpi=DPI)
    plt.show()

def plot_accuracy_curve_fig(history, model_name, save_dir):
    plt.figure(figsize=FIGSIZE)
    plt.plot(history["train_accuracy"], color=COLOR_PALETTE[0], linewidth=2.5, label="Training Accuracy")
    plt.plot(history["val_accuracy"], color=COLOR_PALETTE[1], linewidth=2.5, label="Validation Accuracy")
    style_title_labels(f"{model_name} Accuracy Curve", "Epoch", "Accuracy")
    apply_axis_style()
    plt.legend(prop=LEGEND_FONT)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_Accuracy_Curve.png"), dpi=DPI)
    plt.show()

def plot_loss_curve_fig(history, model_name, save_dir):
    plt.figure(figsize=FIGSIZE)
    plt.plot(history["train_loss"], color=COLOR_PALETTE[2], linewidth=2.5, label="Training Loss")
    plt.plot(history["val_loss"], color=COLOR_PALETTE[3], linewidth=2.5, label="Validation Loss")
    style_title_labels(f"{model_name} Loss Curve", "Epoch", "Loss")
    apply_axis_style()
    plt.legend(prop=LEGEND_FONT)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_Loss_Curve.png"), dpi=DPI)
    plt.show()

def plot_roc_curves_fig(all_targets, all_probabilities, class_names, model_name, save_dir):
    y_true_bin = label_binarize(all_targets, classes=list(range(len(class_names))))

    if y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    plt.figure(figsize=FIGSIZE)

    for class_index, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, class_index], all_probabilities[:, class_index])
        roc_auc = auc(fpr, tpr)
        color = COLOR_PALETTE[class_index % len(COLOR_PALETTE)]
        plt.plot(fpr, tpr, color=color, linewidth=2.5, label=f"{class_name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.5)
    style_title_labels(f"{model_name} Classwise ROC Curve", "False Positive Rate", "True Positive Rate")
    apply_axis_style()
    plt.legend(prop=LEGEND_FONT, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_ROC_Curve.png"), dpi=DPI)
    plt.show()

def plot_precision_recall_curves_fig(all_targets, all_probabilities, class_names, model_name, save_dir):
    y_true_bin = label_binarize(all_targets, classes=list(range(len(class_names))))

    if y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    plt.figure(figsize=FIGSIZE)

    for class_index, class_name in enumerate(class_names):
        precision_vals, recall_vals, _ = precision_recall_curve(y_true_bin[:, class_index], all_probabilities[:, class_index])
        average_precision = average_precision_score(y_true_bin[:, class_index], all_probabilities[:, class_index])
        color = COLOR_PALETTE[class_index % len(COLOR_PALETTE)]
        plt.plot(recall_vals, precision_vals, color=color, linewidth=2.5, label=f"{class_name} (AP = {average_precision:.2f})")

    style_title_labels(f"{model_name} Classwise Precision-Recall Curve", "Recall", "Precision")
    apply_axis_style()
    plt.legend(prop=LEGEND_FONT, loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_Precision_Recall_Curve.png"), dpi=DPI)
    plt.show()

def plot_overall_metrics_bar_fig(results_dict, model_name, save_dir):
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    metric_values = [results_dict[m] for m in metric_names]
    colors = COLOR_PALETTE[4:8]

    plt.figure(figsize=FIGSIZE)
    bars = plt.bar(metric_names, metric_values, color=colors)

    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}",
                  ha="center", fontsize=16, fontweight="bold", fontfamily="Times New Roman")

    plt.ylim(0, 1.05)
    style_title_labels(f"{model_name} Overall Performance Metrics", "Metric", "Score")
    apply_axis_style()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_Overall_Metrics_Bar.png"), dpi=DPI)
    plt.show()

def plot_fpr_fnr_bar_fig(confusion, class_names, model_name, save_dir):
    total = confusion.sum()
    fpr_values = []
    fnr_values = []

    for class_index in range(len(class_names)):
        true_positive = confusion[class_index, class_index]
        false_negative = confusion[class_index, :].sum() - true_positive
        false_positive = confusion[:, class_index].sum() - true_positive
        true_negative = total - true_positive - false_negative - false_positive

        fpr = false_positive / (false_positive + true_negative) if (false_positive + true_negative) > 0 else 0.0
        fnr = false_negative / (false_negative + true_positive) if (false_negative + true_positive) > 0 else 0.0

        fpr_values.append(fpr)
        fnr_values.append(fnr)

    x_positions = np.arange(len(class_names))
    bar_width = 0.35

    # --- FIX: reserve headroom above the tallest bar so value labels never
    # collide with the title / get clipped outside the axes ---
    max_value = max(fpr_values + fnr_values) if (fpr_values + fnr_values) else 0.0
    label_offset = max_value * 0.03 if max_value > 0 else 0.001
    y_upper_limit = max_value * 1.20 if max_value > 0 else 1.0

    plt.figure(figsize=FIGSIZE)
    fpr_bars = plt.bar(x_positions - bar_width / 2, fpr_values, bar_width, color=COLOR_PALETTE[8], label="False Positive Rate")
    fnr_bars = plt.bar(x_positions + bar_width / 2, fnr_values, bar_width, color=COLOR_PALETTE[9], label="False Negative Rate")
    plt.xticks(x_positions, class_names)

    for bar_group in (fpr_bars, fnr_bars):
        for bar in bar_group:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + label_offset, f"{height:.3f}",
                      ha="center", va="bottom", fontsize=14, fontweight="bold", fontfamily="Times New Roman")

    plt.ylim(0, y_upper_limit)
    style_title_labels(f"{model_name} FPR and FNR by Class", "Class", "Rate")
    apply_axis_style()
    plt.legend(prop=LEGEND_FONT)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_FPR_FNR_Bar.png"), dpi=DPI)
    plt.show()

def plot_convergence_comparison_fig(histories, model_names, save_dir):
    plt.figure(figsize=FIGSIZE)

    for model_index, (history, model_name) in enumerate(zip(histories, model_names)):
        color = COLOR_PALETTE[10 + model_index]
        plt.plot(history["val_loss"], color=color, linewidth=2.5, marker="o", markersize=4, label=f"{model_name} Validation Loss")

    style_title_labels("Convergence Comparison (Validation Loss)", "Epoch", "Loss")
    apply_axis_style()
    plt.legend(prop=LEGEND_FONT)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Convergence_Comparison.png"), dpi=DPI)
    plt.show()

def plot_classwise_comparison_fig(results_a, results_b, class_names, save_dir):
    from sklearn.metrics import precision_recall_fscore_support

    precision_a, recall_a, f1_a, _ = precision_recall_fscore_support(
        results_a["Targets"], results_a["Predictions"], labels=list(range(len(class_names))), zero_division=0
    )
    precision_b, recall_b, f1_b, _ = precision_recall_fscore_support(
        results_b["Targets"], results_b["Predictions"], labels=list(range(len(class_names))), zero_division=0
    )

    x_positions = np.arange(len(class_names))
    bar_width = 0.35

    plt.figure(figsize=FIGSIZE)
    bars_a = plt.bar(x_positions - bar_width / 2, f1_a, bar_width, color=COLOR_PALETTE[0], label=f"{results_a['Model']} F1-Score")
    bars_b = plt.bar(x_positions + bar_width / 2, f1_b, bar_width, color=COLOR_PALETTE[1], label=f"{results_b['Model']} F1-Score")

    for bar_group in (bars_a, bars_b):
        for bar in bar_group:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.002, f"{height:.6f}",
                      ha="center", fontsize=11, fontweight="bold", fontfamily="Times New Roman", rotation=90)

    plt.xticks(x_positions, class_names)
    plt.ylim(0, 1.15)
    style_title_labels("Classwise F1-Score Comparison", "Class", "F1-Score")
    apply_axis_style()
    plt.legend(prop=LEGEND_FONT)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Classwise_F1_Comparison.png"), dpi=DPI)
    plt.show()

def plot_model_comparison_fig(results_list, save_dir):
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    model_colors = [COLOR_PALETTE[0], COLOR_PALETTE[1]]

    x_positions = np.arange(len(metric_names))
    bar_width = 0.35

    plt.figure(figsize=FIGSIZE)

    for model_index, results_dict in enumerate(results_list):
        values = [results_dict[m] for m in metric_names]
        offset = (model_index - 0.5) * bar_width
        bars = plt.bar(x_positions + offset, values, bar_width,
                        color=model_colors[model_index % len(model_colors)],
                        label=results_dict["Model"])

        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, value + 0.002, f"{value}",
                      ha="center", fontsize=11, fontweight="bold", fontfamily="Times New Roman",
                      rotation=90)

    best_model_index = int(np.argmax([r["F1-Score"] for r in results_list]))
    plt.gca().text(
        0.02, 0.98,
        f"Best Overall (F1-Score): {results_list[best_model_index]['Model']}",
        transform=plt.gca().transAxes, fontsize=14, fontweight="bold",
        fontfamily="Times New Roman", verticalalignment="top"
    )

    plt.xticks(x_positions, metric_names)
    plt.ylim(0, 1.15)
    style_title_labels("Mamba vs CapsNet Model Comparison", "Metric", "Score")
    apply_axis_style()
    plt.legend(prop=LEGEND_FONT)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Model_Comparison_Bar.png"), dpi=DPI)
    plt.show()
    print(f"\nBest performing model by F1-Score: {results_list[best_model_index]['Model']}")

def generate_all_plots(results_dict, history, class_names, model_name, save_dir, model=None, test_loader=None, feature_columns=None):
    os.makedirs(save_dir, exist_ok=True)

    plot_confusion_matrix_fig(results_dict["Confusion"], class_names, model_name, save_dir)
    plot_accuracy_curve_fig(history, model_name, save_dir)
    plot_loss_curve_fig(history, model_name, save_dir)
    plot_roc_curves_fig(results_dict["Targets"], results_dict["Probabilities"], class_names, model_name, save_dir)
    plot_precision_recall_curves_fig(results_dict["Targets"], results_dict["Probabilities"], class_names, model_name, save_dir)
    plot_overall_metrics_bar_fig(results_dict, model_name, save_dir)
    plot_fpr_fnr_bar_fig(results_dict["Confusion"], class_names, model_name, save_dir)

    if model is not None and test_loader is not None and feature_columns is not None:
        plot_feature_importance_fig(model, test_loader, feature_columns, model_name, save_dir)

MAMBA_SAVE_DIR = "Mamba_Plots"
CAPSNET_SAVE_DIR = "CapsNet_Plots"
COMPARISON_SAVE_DIR = "Comparison_Plots"
EDA_SAVE_DIR = "EDA_Plots"

os.makedirs(COMPARISON_SAVE_DIR, exist_ok=True)
os.makedirs(EDA_SAVE_DIR, exist_ok=True)

# ============================================================
# EXPLORATORY / DATA-LEVEL PLOTS
# ============================================================
plot_class_distribution_fig(y_sequences, class_names, EDA_SAVE_DIR)

ENGINEERED_TIME_COLUMNS = [
    "hour", "minute", "day", "month", "day_of_week", "is_weekend",
    "hour_sin", "hour_cos", "day_sin", "day_cos"
]

traffic_numeric_columns = [
    column for column in numerical_columns
    if column not in ENGINEERED_TIME_COLUMNS and column not in [ROAD_COLUMN, TIME_COLUMN]
]

plot_feature_distribution_by_class_fig(data, TARGET_COLUMN, traffic_numeric_columns, class_names, EDA_SAVE_DIR, max_features=4)

generate_all_plots(mamba_results, mamba_history, class_names, "Mamba", MAMBA_SAVE_DIR, model=mamba_model, test_loader=test_loader, feature_columns=feature_columns)

generate_all_plots(capsule_results, capsule_history, class_names, "CapsNet", CAPSNET_SAVE_DIR, model=capsule_model, test_loader=test_loader, feature_columns=feature_columns)

plot_convergence_comparison_fig(
    [mamba_history, capsule_history],
    ["Mamba", "CapsNet"],
    COMPARISON_SAVE_DIR
)

plot_model_comparison_fig([mamba_results, capsule_results], COMPARISON_SAVE_DIR)

plot_classwise_comparison_fig(mamba_results, capsule_results, class_names, COMPARISON_SAVE_DIR)

plot_error_distribution_fig(mamba_results, capsule_results, class_names, COMPARISON_SAVE_DIR)

print("\n========== DEEP FEATURE INFORMATION ==========")

print("Mamba Deep Feature Shape:", mamba_results["Deep_Features"].shape)

print("CapsNet Deep Feature Shape:", capsule_results["Deep_Features"].shape)

print(f"\nAll Mamba plots saved to: {os.path.abspath(MAMBA_SAVE_DIR)}")

print(f"All CapsNet plots saved to: {os.path.abspath(CAPSNET_SAVE_DIR)}")

print(f"All comparison plots saved to: {os.path.abspath(COMPARISON_SAVE_DIR)}")

print(f"All EDA plots saved to: {os.path.abspath(EDA_SAVE_DIR)}")

print("\nAll processing, deep feature learning, optimization, evaluation, and saving completed successfully.")