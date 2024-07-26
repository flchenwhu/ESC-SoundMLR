import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from thop import profile
from torch.utils.tensorboard import SummaryWriter  # 导入 tensorboard 模块

import config
from loss_fn.hybrid_loss import HybridLoss
from models import model_classifier
from models import model_projection
from utils.my_utils import EarlyStopping, WarmUpExponentialLR

if config.ESC_50:
    import dataset_ESC50 as dataset
elif config.ESC51:
    import dataset_ESC51 as dataset
elif config.US8K:
    import dataset_US8K as dataset




class SpectralPoolingAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(SpectralPoolingAttention, self).__init__()
        mid_channels = max(out_channels // reduction, 1)  # 确保中间通道数至少为1
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, channels, frequency, time)
        y = torch.mean(x, dim=-1)  # (batch_size, channels, frequency)
        y = self.conv1(y.unsqueeze(-1))
        y = self.relu(y)
        y = self.conv2(y)
        y = self.sigmoid(y)  # (batch_size, channels, frequency, 1)
        x = x * y.expand_as(x)

        return x



class my_model(nn.Module):
    def __init__(self, model_type, in_channels=3, out_channels=3, factors=0.75, reduction=16):
        super(my_model, self).__init__()
        # Assuming `build_model` function is defined elsewhere and loads a pretrained model
        self.pretrain = build_model(model_type)

        # Replace the final layer of the pretrained model with an Identity layer to use it as a feature extractor
        if hasattr(self.pretrain, 'fc'):
            self.pretrain.fc = nn.Identity()  # Replace fully connected layer
        elif hasattr(self.pretrain, 'classifier'):
            self.pretrain.classifier[-1] = nn.Identity()  # Replace classifier layer

        self.se = SpectralPoolingAttention(in_channels=3, out_channels=3, reduction=16)


        # Initialize the Pooling_Layer with the provided factors and reduction
        # self.pooling_layer = Pooling_Layer(in_channels=in_channels, factor=factors, reduction=reduction)
        # A convolutional layer to adjust the output channel dimensions as needed
        # self.post_pooling_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):

        x = self.se(x)

        # x = self.pooling_layer(x)
        # x = self.post_pooling_conv(x)

        x = self.pretrain(x)
        return x

def calculate_flops_and_params(model, input_size):
    model.eval()
    input_tensor = torch.randn(1, *input_size)
    flops, params = profile(model, inputs=(input_tensor,))
    return flops, params




def build_model(model_type):
    """构建模型"""
    if model_type == "mbv2":
        model = torchvision.models.mobilenet_v2(pretrained=True)  # 使用torchvision提供的预训练模型
    elif model_type == "resnet50":
        model = torchvision.models.resnet50(pretrained=True)
    elif model_type == "resnet18":
        model = torchvision.models.resnet18(pretrained=True)
    elif model_type == "vgg16":
        model = torchvision.models.vgg16(pretrained=True)
    elif model_type == "mbv3":
        model = torchvision.models.mobilenet_v3_large(pretrained=True)
    elif model_type == "resnet34":
        model = torchvision.models.resnet34(pretrained=True)
    else:
        raise Exception("Error:{}".format(model_type))

    return model



use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


model_type = 'mbv2'
input_size = (3, 128, 431)  # esc50/esc51
# input_size = (3, 128, 344) #us8k
model = my_model(model_type)

flops, params = calculate_flops_and_params(model, input_size)

print("Flops: {:.4f} M".format(flops / 1e6))
print("Params: {:.4f} M".format(params / 1e6))
model = model.to(device)




# creating a folder to save the reports and models
main_path = f'results/hybridLoss_MLA_{model_type}_ESC51'
classifier_path = main_path + '/' + 'classifier'
# projection_layer_path = main_path + '/' + 'projection_layer'



def hotEncoder(v):
    ret_vec = torch.zeros(v.shape[0], config.class_numbers).to(device)
    for s in range(v.shape[0]):
        ret_vec[s][v[s]] = 1
    return ret_vec
from sklearn.metrics import precision_score, recall_score, f1_score

def compute_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)#macro
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return precision, recall, f1



def train_hybrid():
    train_loader, val_loader = dataset.create_generators()
    projection_layer = model_projection.ProjectionModel().to(device)
    classifier = model_classifier.Classifier().to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(projection_layer.parameters()) + list(classifier.parameters()),
        lr=config.lr, weight_decay=1e-3
    )
    loss_fn = HybridLoss(
        alpha=config.alpha,
        beta=config.beta,
        temperature=config.temperature,
        adasp_temp=0.04,
        loss_type="adasp",
    ).to(device)

    scheduler = WarmUpExponentialLR(optimizer, cold_epochs=0, warm_epochs=config.warm_epochs, gamma=config.gamma)
    num_epochs = 800
    mainModel_stopping = EarlyStopping(patience=200, verbose=True, log_path=main_path)
    classifier_stopping = EarlyStopping(patience=200, verbose=False, log_path=classifier_path
                                        )

    print('*****')
    print('HYBRID')
    print('alpha is {}'.format(config.alpha))
    print('beta is {}'.format(config.beta))

    if config.ESC_50:
        print('ESC_50')
        print('train folds are {} and test fold is {}'.format(config.train_folds, config.test_fold))
    elif config.US8K:
        print('US8K')
        print('train folds are {} and test fold is {}'.format(config.train_folds, config.test_fold) )
    elif config.ESC51:
        print('ESC_51:')
        print('train folds are {} and test fold is {}'.format(config.train_folds, config.test_fold))

    print('Freq mask number {} and length {}, and time mask number {} and length is{}'.format(config.freq_masks,
                                                                                              config.freq_masks_width,
                                                                                              config.time_masks,
                                                                                              config.time_masks_width))

    print('*****')


    best_acc = 0
    best_epoch = 0
    best_precision = 0
    best_recall = 0
    best_f1 = 0

    # Initialize lists to collect metrics across epochs
    all_train_accuracies = []
    all_val_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1_scores = []

    for epoch in range(num_epochs):
        print('\n Learning Rate: ' + str(optimizer.param_groups[0]["lr"]))
        model.train()
        projection_layer.train()
        classifier.train()

        train_loss = []
        train_corrects = 0
        train_samples_count = 0

        for i, (_, x, label) in enumerate(train_loader):
            optimizer.zero_grad()

            x = x.float().to(device)
            label = label.to(device).unsqueeze(1)
            label_vec = hotEncoder(label)

            y_rep = model(x)
            y_rep = F.normalize(y_rep, dim=0)
            y_proj = projection_layer(y_rep)
            y_proj = F.normalize(y_proj, dim=0)
            y_pred = classifier(y_rep)

            loss = loss_fn(y_proj, y_pred, label, label_vec)
            loss.backward()

            train_loss.append(loss.item())

            optimizer.step()

            train_corrects += (torch.argmax(y_pred, dim=1) == torch.argmax(label_vec, dim=1)).sum().item()
            train_samples_count += x.shape[0]

            if i % 10 == 0:
                print(f"Epoch: {epoch + 1}/{num_epochs} Batch {i}/{len(train_loader)} "
                      f"Loss: {np.mean(train_loss):.4f}")

        train_acc = train_corrects / train_samples_count
        all_train_accuracies.append(train_acc)

        val_loss = []
        val_corrects = 0
        val_samples_count = 0

        model.eval()
        projection_layer.eval()
        classifier.eval()

        with torch.no_grad():
            for _, val_x, val_label in val_loader:
                val_x = val_x.float().to(device)
                label = val_label.to(device).unsqueeze(1)
                label_vec = hotEncoder(label)
                y_rep = model(val_x)
                y_rep = F.normalize(y_rep, dim=0)
                y_proj = projection_layer(y_rep)
                y_proj = F.normalize(y_proj, dim=0)
                y_pred = classifier(y_rep)

                loss = loss_fn(y_proj, y_pred, label, label_vec)

                val_loss.append(loss.item())

                val_corrects += (torch.argmax(y_pred, dim=1) == torch.argmax(label_vec, dim=1)).sum().item()
                val_samples_count += val_x.shape[0]

                val_preds = torch.argmax(y_pred, dim=1).cpu().numpy()
                val_labels = torch.argmax(label_vec, dim=1).cpu().numpy()
                val_precision, val_recall, val_f1 = compute_metrics(val_labels, val_preds)
                all_precisions.append(val_precision)
                all_recalls.append(val_recall)
                all_f1_scores.append(val_f1)

        val_acc = val_corrects / val_samples_count
        all_val_accuracies.append(val_acc)

        scheduler.step()

        print(
            f'Test {epoch + 1} val_loss is {np.mean(val_loss):.4f} '
            f'train_acc is {train_acc:.4f} and val_acc is {val_acc:.4f}')
        print(f'Validation Precision: {val_precision:.4f}')
        print(f'Validation Recall: {val_recall:.4f}')
        print(f'Validation F1-Score: {val_f1:.4f}')


        mainModel_stopping(-val_acc, model, epoch + 1)
        classifier_stopping(-val_acc, classifier, epoch + 1)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1

            best_precision = val_precision
            best_recall = val_recall
            best_f1 = val_f1


        print(f'Best Precision: {best_precision:.4f}')
        print(f'Best Recall: {best_recall:.4f}')
        print(f'Best F1-Score: {best_f1:.4f}')
        print('Best accuracy: {:.4f} at epoch {}'.format(best_acc, best_epoch))
        if mainModel_stopping.early_stop:
            print("Early stopping")
            break


    # Calculate and print the mean and standard deviation for all metrics
    mean_train_acc, std_train_acc = np.mean(all_train_accuracies), np.std(all_train_accuracies)
    mean_val_acc, std_val_acc = np.mean(all_val_accuracies), np.std(all_val_accuracies)
    mean_precision, std_precision = np.mean(all_precisions), np.std(all_precisions)
    mean_recall, std_recall = np.mean(all_recalls), np.std(all_recalls)
    mean_f1, std_f1 = np.mean(all_f1_scores), np.std(all_f1_scores)

    print(f'Average Training Accuracy: {mean_train_acc:.4f}'
          f' ± {std_train_acc:.4f}')
    print(f'Average Validation Accuracy: {mean_val_acc:.4f} ± {std_val_acc:.4f}')
    print(f'Average Precision: {mean_precision:.4f} ± {std_precision:.4f}')
    print(f'Average Recall: {mean_recall:.4f} ± {std_recall:.4f}')
    print(f'Average F1-Score: {mean_f1:.4f} ± {std_f1:.4f}')








if __name__ == "__main__":
    train_hybrid()
