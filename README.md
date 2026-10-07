# 🥔 木薯横截面坏死分割识别（UNet）

基于 PyTorch UNet 语义分割网络的木薯（Cassava）横截面坏死病斑自动识别与分级系统。

## 📖 项目简介

本项目利用深度学习语义分割技术，对木薯横截面图像中的**健康薯肉**与**坏死病斑**区域进行像素级分割，并基于坏死占比自动判定病害严重程度等级，为木薯品质检测与病害防控提供自动化解决方案。

### 核心功能

- 🔍 **像素级语义分割**：基于 UNet 网络对木薯横截面图像进行三分割（背景 / 健康薯肉 / 坏死病斑）
- 📊 **坏死占比计算**：自动计算坏死病斑占有效组织面积的比例
- 🏷️ **病害等级分类**：根据坏死占比进行三级分级（轻度 / 中度 / 重度）
- 🖼️ **可视化分析**：提供训练过程、预测结果、数据增强等多种可视化

## 🧠 模型架构

采用经典 **U-Net** 编码器-解码器架构，带有跳跃连接：

```
输入 (3×256×256) → Encoder → Bottleneck → Decoder → 输出 (3×256×256)
                     ↓↑              ↓↑              ↓↑
              Skip Connections  Skip Connections  Skip Connections
```

- **编码器**：4 层下采样，特征通道数 [64, 128, 256, 512]
- **解码器**：4 层上采样，对称跳跃连接
- **基础模块**：DoubleConv（Conv2d → BatchNorm → ReLU）× 2

## 🏷️ 标签体系

### 语义分割类别（三类）

| 原始灰度值 | 类别 | 映射值 | 可视化颜色 |
|:---:|:---:|:---:|:---:|
| 0 | 背景 | 0 | 黑色 |
| 75 | 健康薯肉 | 1 | 绿色 |
| 38 | 坏死病斑 | 2 | 红色 |

### 坏死严重程度分级（三级）

| 坏死占比 | 等级 | 英文标识 |
|:---:|:---:|:---:|
| ≤ 5% | 轻度（一级） | Mild (L1) |
| > 5% 且 ≤ 20% | 中度（二级） | Moderate (L2) |
| > 20% | 重度（三级） | Severe (L3) |

**坏死占比计算公式：**

$$\text{坏死占比} = \frac{\text{坏死病斑像素数}}{\text{健康像素数} + \text{坏死像素数}} \times 100\%$$

## 📂 项目结构

```
木薯横截面坏死分割识别/
├── 数据集/
│   └── labeled/
│       ├── images/          # 原始木薯横截面图像（1036 张 .jpg）
│       └── masks/           # 对应分割掩码（1036 张 .png）
├── cassava_unet（含源代码和测试内容）.ipynb   # 主 Notebook（完整源码+测试）
├── cassava_unet（含源代码和测试内容）.pdf     # Notebook PDF 导出
├── cassava_unet(含源代码和测试文件).pdf       # Notebook PDF 导出（备）
├── cassava_notebook_cn.pdf                   # 项目说明文档
├── 测试文档.pdf                               # 测试报告
├── September_2018.zip                        # 原始数据集压缩包
└── README.md
```

## 📊 数据集

- **数据量**：1036 对图像-掩码对
- **训练集**：828 张（80%）
- **验证集**：208 张（20%）
- **图像格式**：原图 `.jpg`，掩码 `.png`（单通道灰度图）
- **图像尺寸**：统一 Resize 至 256×256

## ⚙️ 训练配置

| 参数 | 值 |
|:---|:---|
| 模型 | UNet |
| 输入尺寸 | 256 × 256 |
| 批次大小 | 4 |
| 训练轮次 | 150 |
| 学习率 | 1e-4 |
| 优化器 | Adam（weight_decay=1e-5） |
| 损失函数 | CE + Dice Loss（各 0.5 权重） |
| 分割类别数 | 3（背景 / 健康 / 坏死） |
| 随机种子 | 42 |

### 数据增强（训练集）

- `Resize(256, 256)`
- `HorizontalFlip(p=0.5)`
- `VerticalFlip(p=0.3)`
- `RandomRotate90(p=0.3)`
- `ShiftScaleRotate(shift=0.05, scale=0.1, rotate=15, p=0.3)`
- `Normalize(mean, std)` + `ToTensorV2()`

### 数据增强（验证集）

- `Resize(256, 256)`
- `Normalize(mean, std)` + `ToTensorV2()`

## 🔧 环境依赖

```
Python >= 3.8
PyTorch >= 1.10
torchvision
opencv-python (cv2)
numpy
pandas
matplotlib
scikit-learn
albumentations
tqdm
```

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/<your-username>/木薯横截面坏死分割识别.git
cd 木薯横截面坏死分割识别
```

### 2. 安装依赖

```bash
pip install torch torchvision opencv-python numpy pandas matplotlib scikit-learn albumentations tqdm
```

### 3. 运行 Notebook

使用 Jupyter Notebook 或 JupyterLab 打开主文件：

```bash
jupyter notebook "cassava_unet（含源代码和测试内容）.ipynb"
```

按顺序执行各 Cell 即可完成数据处理、模型训练、验证与预测。

### 4. 训练产出

训练完成后，最佳模型权重保存为：

```
best_unet_cassava.pth
```

## 📈 Notebook 内容结构

| 章节 | 内容 |
|:---:|:---|
| 一 | 数据加载与预处理（路径匹配、掩码重映射、数据集划分） |
| 二 | 数据探索与可视化（灰度验证、RGB 分析、坏死占比统计、数据增强演示） |
| 三 | 数据集与 DataLoader 构建（CassavaNecrosisDataset、增强管线） |
| 四 | UNet 模型定义与训练（模型构建、Dice Loss、训练循环、早停保存） |
| 五 | 预测结果与病害分级（模型加载、推理函数、预测可视化、分级输出） |

## 📝 许可证

本项目仅供学习与研究使用。

## 🙏 致谢

- 数据集来源：September 2018 Cassava Disease Dataset
- 模型架构参考：Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015
