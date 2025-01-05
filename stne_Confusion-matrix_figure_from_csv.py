import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

# 自定义类名列表
######################################ESC51
class_names = [
    "Dog", "Rain", "Crying baby", "Door knock", "Helicopter", "Rooster", "Sea waves",
    "Sneezing", "Mouse click", "Chainsaw", "Pig", "Crackling fire", "Clapping",
    "Keyboard typing", "Siren", "Cow", "Crickets", "Breathing", "Door, wood creaks",
    "Car horn", "Frog", "Chirping birds", "Coughing", "Can opening", "Engine", "Cat",
    "Water drops", "Footsteps", "Washing machine", "Train", "Hen", "Wind", "Laughing",
    "Vacuum cleaner", "Church bells", "Insects (flying)", "Pouring water",
    "Brushing teeth", "Clock alarm", "Airplane", "Sheep", "Toilet flush", "Snoring",
    "Clock tick", "Fireworks", "Crow", "Thunderstorm", "Drinking, sipping",
    "Glass breaking", "Hand saw", "UAV"
]

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
# class_names = [
#     "Air conditioner", "Car Horn", "Children playing", "Dog Bark", "Drilling", "Engine ldling", "Gun shot",
#     "Jackhammer", "Siren", "Street music"
# ]

# 用于绘制 t-SNE 可视化图的函数
def plot_tsne(csv_file):
    # 读取 t-SNE 数据
    df = pd.read_csv(csv_file)
    
    # 提取 t-SNE 的两个维度数据和真实标签
    tsne_1 = df['tsne_1']
    tsne_2 = df['tsne_2']
    true_labels = df['true_label']
    
    # 使用自定义的类名
    # class_names_dict = {i: name for i, name in enumerate(class_names)}
    colors = plt.cm.rainbow(np.linspace(0, 1, len(class_names)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', 'd', 'P', 'X']

    # 创建 t-SNE 图
    plt.figure(figsize=(18, 16))  # 调整图像宽度
    ax = plt.gca()

    for i, class_name in enumerate(class_names):
        # 这里修正了：使用 class_name 而不是 i
        mask = true_labels == class_name  # 根据类名进行过滤
        ax.scatter(tsne_1[mask], tsne_2[mask], c=[colors[i]], label=class_name,
                   marker=markers[i % len(markers)], s=50, alpha=0.7)  # 增大标记大小

    # 设置坐标轴和标题
    ax.set_xlabel('t-SNE feature 1', fontsize=18)
    ax.set_ylabel('t-SNE feature 2', fontsize=18)
    ax.grid(True, linestyle='--', linewidth=1.5, alpha=0.7)
    # 调整坐标轴刻度值的大小
    ax.tick_params(axis='both', which='major', labelsize=17)

    # 调整图例并增大字体
    plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=6, fontsize=15)
    plt.tight_layout()

    # 输出文件路径
    output_file = os.path.splitext(csv_file)[0] + '_tsne.png'

    # 保存图片
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    # plt.show()

# 用于绘制混淆矩阵的函数
def plot_confusion_matrix(csv_file):
    # 读取混淆矩阵数据
    df = pd.read_csv(csv_file, index_col=0)
    cm = df.values

    # 创建混淆矩阵图，调整图像尺寸
    fig_cm = plt.figure(figsize=(28, 17))  # 调整图像大小
    # ax = plt.gca()
    ax = fig_cm.add_subplot(111)

    # 绘制热力图，设置白色背景和黑色框
    # sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', annot_kws={'size': 13, 'weight': 'bold', 'color': 'black'},
    #             xticklabels=df.columns, yticklabels=df.index, ax=ax, cbar=True,
    #             cbar_kws={'label': 'Normalized Frequency', 'shrink': 0.6},  # 调整颜色棒宽度比例
    #             linewidths=1, linecolor='black')
    # 绘制热力图，设置白色背景和黑色框
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', annot_kws={'size': 13, 'color': 'black'},
                xticklabels=df.columns, yticklabels=df.index, ax=ax, cbar=True,
                # cbar_kws={'label': 'Normalized frequency', 'shrink': 0.8},  #  调整颜色条的长度
                linewidths=0.7, linecolor='black', square=False, vmin=-0.04, vmax=1)

    # # 针对对角线的数值修改字体颜色
    # for i in range(len(df.columns)):
    #     for j in range(len(df.columns)):
    #         text = ax.texts[i * len(df.columns) + j]  # 获取每个单元格的文本对象
    #         if i == j:  # 对角线上的元素
    #             text.set_color('lightgray')  # 设置对角线数值为浅灰白色   
    # 针对对角线的数值修改字体颜色
    for i in range(len(df.columns)):
        for j in range(len(df.columns)):
            text = ax.texts[i * len(df.columns) + j]  # 获取每个单元格的文本对象
            if i == j:  # 对角线上的元素
                value = float(text.get_text())  # 获取单元格的值
                if value > 0.50:
                    text.set_color('lightgray')  # 设置对角线数值为浅灰白色
                else:
                    text.set_color('darkslategray')  # 或者使用'darkslategray'   'lightgray'


    # 设置标题和坐标轴标签字体
    ax.set_xlabel('Predicted label', fontsize=20)
    ax.set_ylabel('True label', fontsize=20)

    # 设置标签字体大小和旋转
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=15)
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", fontsize=15)

    # 调整颜色条相关的刻度和标签字体
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)  # 设置刻度标签的字体大小
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])  # 调整刻度的显示范围
    cbar.set_label('Normalized frequency', fontsize=15)  # 设置颜色条标签的字体大小
    plt.tight_layout() # 会调整图形的整体布局，特别是在有多个子图时
    cbar.ax.set_position([0.84, 0.02, 0.013, 1])  # 设置颜色条的宽度和位置
    # 用来调整 颜色条 (colorbar) 的 位置和尺寸 的。set_position() 方法接受一个包含四个元素的列表 
    # [left, bottom, width, height]，这四个值分别代表颜色条的位置和大小。

    # 不显示为 0 的数据
    for text in ax.texts:
        if text.get_text() == "0.00":
            text.set_text("")  # 将 "0" 清空

    # 输出文件路径
    output_file = os.path.splitext(csv_file)[0] + '_confusion_matrix.png'

    # 保存图片
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    # plt.show()

# CSV 文件路径定义（方便替换）
tsne_csv = './Hybrid_mbv2_ESC51_tsne.csv'  # 更新此路径为您的 CSV 文件路径
cm_csv = './Hybrid_mbv2_ESC51_confusion_matrix.csv'  # 更新此路径为您的 CSV 文件路径

# 绘制 t-SNE 图和混淆矩阵图
plot_tsne(tsne_csv)
plot_confusion_matrix(cm_csv)
