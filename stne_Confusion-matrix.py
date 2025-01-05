import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import os

# Assuming these are in separate files
import config
from SimPFs.pooling import Pooling_Layer

if config.ESC_50:
    import dataset_ESC50 as dataset
elif config.ESC51:
    import dataset_ESC51 as dataset
elif config.US8K:
    import dataset_US8K as dataset
from models.model_classifier import Classifier
from models.model_projection import ProjectionModel

# CUDA setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SpectralPoolingAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(SpectralPoolingAttention, self).__init__()
        mid_channels = max(out_channels // reduction, 1)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = torch.mean(x, dim=-1)
        y = self.conv1(y.unsqueeze(-1))
        y = self.relu(y)
        y = self.conv2(y)
        y = self.sigmoid(y)
        x = x * y.expand_as(x)
        return x

class my_model(nn.Module):
    def __init__(self, model_type, in_channels=3, out_channels=3, factors=0.75,  reduction=16):
        super(my_model, self).__init__()
        self.pretrain = self.build_model(model_type)
        if hasattr(self.pretrain, 'fc'):
            self.pretrain.fc = nn.Identity()
        elif hasattr(self.pretrain, 'classifier'):
            self.pretrain.classifier[-1] = nn.Identity()
        self.se = SpectralPoolingAttention(in_channels=3, out_channels=3, reduction=16)
        self.pooling_layer = Pooling_Layer(in_channels=in_channels, factor=factors, reduction=reduction)
        self.post_pooling_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def build_model(self, model_type):
        if model_type == "mbv2":
            return torchvision.models.mobilenet_v2(pretrained=True)
        elif model_type == "resnet50":
            return torchvision.models.resnet50(pretrained=True)
        elif model_type == "resnet18":
            return torchvision.models.resnet18(pretrained=True)
        elif model_type == "vgg16":
            return torchvision.models.vgg16(pretrained=True)
        elif model_type == "mbv3":
            return torchvision.models.mobilenet_v3_large(pretrained=True)
        elif model_type == "resnet34":
            return torchvision.models.resnet34(pretrained=True)
        else:
            raise Exception(f"Error: Unsupported model type {model_type}")

    def forward(self, x):

        x = self.se(x)
        # x = self.pooling_layer(x)
        # x = self.post_pooling_conv(x)
        features = self.pretrain(x)
        return features

def plot_tsne_and_confusion_matrix(features, true_labels, predicted_labels, title):
    # Class names for ESC51 (adjust as needed for other datasets)
    # class_names = [
    #     "Dog", "Rain", "Crying baby", "Door knock", "Helicopter", "Rooster", "Sea waves",
    #     "Sneezing", "Mouse click", "Chainsaw", "Pig", "Crackling fire", "Clapping",
    #     "Keyboard typing", "Siren", "Cow", "Crickets", "Breathing", "Door, wood creaks",
    #     "Car horn", "Frog", "Chirping birds", "Coughing", "Can opening", "Engine", "Cat",
    #     "Water drops", "Footsteps", "Washing machine", "Train", "Hen", "Wind", "Laughing",
    #     "Vacuum cleaner", "Church bells", "Insects (flying)", "Pouring water",
    #     "Brushing teeth", "Clock alarm", "Airplane", "Sheep", "Toilet flush", "Snoring",
    #     "Clock tick", "Fireworks", "Crow", "Thunderstorm", "Drinking, sipping",
    #     "Glass breaking", "Hand saw", "Uav"
    # ]

    ######################################ESC50
    # class_names = [
    #     "Dog", "Rain", "Crying baby", "Door knock", "Helicopter", "Rooster", "Sea waves",
    #     "Sneezing", "Mouse click", "Chainsaw", "Pig", "Crackling fire", "Clapping",
    #     "Keyboard typing", "Siren", "Cow", "Crickets", "Breathing", "Door, wood creaks",
    #     "Car horn", "Frog", "Chirping birds", "Coughing", "Can opening", "Engine", "Cat",
    #     "Water drops", "Footsteps", "Washing machine", "Train", "Hen", "Wind", "Laughing",
    #     "Vacuum cleaner", "Church bells", "Insects (flying)", "Pouring water",
    #     "Brushing teeth", "Clock alarm", "Airplane", "Sheep", "Toilet flush", "Snoring",
    #     "Clock tick", "Fireworks", "Crow", "Thunderstorm", "Drinking, sipping",
    #     "Glass breaking", "Hand saw"
    # ]

    # ######################################US8K
    class_names = [
        "Air conditioner", "Car Horn", "Children playing", "Dog Bark", "Drilling", "Engine ldling", "Gun shot",
        "Jackhammer", "Siren", "Street music"
    ]

    # t-SNE visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_results = tsne.fit_transform(features)

    # t-SNE plot
    fig_tsne = plt.figure(figsize=(20, 15))
    ax_tsne = fig_tsne.add_subplot(111)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(class_names)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', 'd', 'P', 'X']

    for i, class_name in enumerate(class_names):
        mask = true_labels == i
        ax_tsne.scatter(
            tsne_results[mask, 0], tsne_results[mask, 1],
            c=[colors[i]], label=class_name,
            marker=markers[i % len(markers)], s=30, alpha=0.6
        )

    ax_tsne.set_title('t-SNE Visualization', fontsize=20)
    ax_tsne.set_xlabel('t-SNE feature 1', fontsize=14)
    ax_tsne.set_ylabel('t-SNE feature 2', fontsize=14)
    ax_tsne.grid(True, linestyle='--', alpha=0.7)

    plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=6, fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{title}_tsne.png', dpi=600, bbox_inches='tight')
    plt.close(fig_tsne)

    # Confusion matrix
    fig_cm = plt.figure(figsize=(22, 18))
    ax_cm = fig_cm.add_subplot(111)
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=class_names, yticklabels=class_names, ax=ax_cm,
                cbar=True, cbar_kws={'label': 'Normalized Frequency'})

    ax_cm.set_title('Confusion Matrix', fontsize=24)
    ax_cm.set_xlabel('Predicted label', fontsize=18)
    ax_cm.set_ylabel('True label', fontsize=18)

    plt.setp(ax_cm.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=10)
    plt.setp(ax_cm.get_yticklabels(), rotation=0, ha="right", fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{title}_confusion_matrix.png', dpi=600, bbox_inches='tight')
    plt.close(fig_cm)

    # Save data to CSV
    df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(features.shape[1])])
    df['true_label'] = [class_names[i] for i in true_labels]
    df['predicted_label'] = [class_names[i] for i in predicted_labels]
    df['tsne_1'] = tsne_results[:, 0]
    df['tsne_2'] = tsne_results[:, 1]
    df.to_csv(f'{title}_data.csv', index=False)

    # Save confusion matrix to CSV
    df_cm = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)
    df_cm.to_csv(f'{title}_confusion_matrix.csv')


def generate_visualizations(model_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = my_model(model_type).to(device)
    projection_layer = ProjectionModel().to(device)
    classifier = Classifier().to(device)

    # 加载保存的模型状态
    best_model_path = f'results/Hybrid_mbv2_US8K/best_model_{model_type}.pth'
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    projection_layer.load_state_dict(checkpoint['projection'])
    classifier.load_state_dict(checkpoint['classifier'])

    model.eval()
    projection_layer.eval()
    classifier.eval()

    # 创建数据加载器
    _, val_loader = dataset.create_generators()

    all_features = []
    all_true_labels = []
    all_predicted_labels = []

    with torch.no_grad():
        for _, x, label in tqdm(val_loader, desc="Processing validation data"):
            x = x.float().to(device)
            label = label.to(device)

            features = model(x)
            y_pred = classifier(features)

            all_features.append(features.cpu().numpy())
            all_true_labels.append(label.cpu().numpy())
            all_predicted_labels.append(torch.argmax(y_pred, dim=1).cpu().numpy())

    # 连接所有数据
    features = np.concatenate(all_features)
    true_labels = np.concatenate(all_true_labels)
    predicted_labels = np.concatenate(all_predicted_labels)

    # 生成图像
    plot_tsne_and_confusion_matrix(features, true_labels, predicted_labels, f'results/Hybrid_mbv2_US8K/{model_type}_final_evaluation')

    print("t-SNE plot and confusion matrix have been generated.")


if __name__ == "__main__":
    model_type = 'mbv2'
    generate_visualizations(model_type)