# 基于 YOLOv8 的火龙果茎部病害目标检测模型

本仓库发布的是一个基于 YOLOv8 训练的火龙果茎部病害目标检测模型，用于识别火龙果茎部常见病害目标，适合病害监测、田间巡检、智慧农业识别服务和后续部署开发。

## 项目概述

该模型面向火龙果茎部病害检测任务，采用 Ultralytics YOLOv8 目标检测框架进行训练与导出，当前仓库包含：

- 最佳训练权重 `weights/best.pt`
- ONNX 部署模型 `weights/best.onnx`
- 训练日志与评估结果 `results.csv`
- 训练曲线、PR 曲线、F1 曲线、混淆矩阵
- 验证集预测可视化样例

![image](https://github.com/jikekei/dragon-fruit-stem-disease-yolov8/blob/26614e1c84be6323c04da3ca358522de277d4ab8/val_batch0_labels.jpg)

## 检测类别

| ID | 类别 | 中文名 | 典型症状 |
|----|------|--------|----------|
| 0 | Brown_Stem_Spot | 褐斑病 | 茎部褐色斑点 |
| 1 | Anthracnose | 炭疽病 | 红褐色圆形病斑 |
| 2 | Gray_Blight | 灰斑病 | 灰白色不规则斑块 |
| 3 | Soft_Rot | 软腐病 | 水渍状软化腐烂 |
| 4 | Stem_Canker | 溃疡病 | 凹陷斑点，后期隆起 |

## 模型亮点

- 基于 YOLOv8 进行火龙果茎部病害目标检测
- 支持 PyTorch 权重与 ONNX 模型两种使用方式
- 保留完整训练结果图，便于论文、比赛或项目汇报
- 适合作为农业病害识别系统的基础检测模型

## 效果概览

本次训练共进行 `50` 个 epoch，输入尺寸为 `512`，批大小为 `8`。

验证集最佳结果如下（以 `mAP50-95` 最优 epoch 统计）：

| 指标 | 数值 |
| --- | ---: |
| Best Epoch | 45 |
| Precision | 0.6908 |
| Recall | 0.6410 |
| mAP@50 | 0.6651 |
| mAP@50:95 | 0.3972 |

最终第 `50` 个 epoch 的结果为：

| 指标 | 数值 |
| --- | ---: |
| Precision | 0.5858 |
| Recall | 0.6762 |
| mAP@50 | 0.6652 |
| mAP@50:95 | 0.3907 |

## 训练配置

| 参数 | 值 |
| --- | --- |
| Task | detect |
| Framework | Ultralytics YOLOv8 |
| Epochs | 50 |
| Image Size | 512 |
| Batch Size | 8 |
| Optimizer | AdamW |
| Initial LR | 0.001 |
| Final LR Factor | 0.0001 |
| Cosine LR | true |
| Warmup Epochs | 5 |
| Device | CUDA:0 |
| Mixed Precision | true |

数据增强配置摘要：

- `degrees=5.0`
- `translate=0.1`
- `scale=0.5`
- `shear=2.0`
- `fliplr=0.5`
- `mosaic=1.0`
- `erasing=0.3`

## 仓库内容

```text
train_optimized/
├── README.md
├── args.yaml
├── results.csv
├── results.png
├── BoxPR_curve.png
├── BoxP_curve.png
├── BoxR_curve.png
├── BoxF1_curve.png
├── confusion_matrix.png
├── confusion_matrix_normalized.png
├── val_batch0_pred.jpg
├── val_batch1_pred.jpg
├── val_batch2_pred.jpg
└── weights/
    ├── best.pt
    ├── best.onnx
    └── last.pt
```

## 快速开始

### 1. 安装依赖

```bash
pip install ultralytics onnxruntime
```

### 2. 使用 PyTorch 权重推理

```python
from ultralytics import YOLO

model = YOLO("weights/best.pt")
results = model.predict(source="demo.jpg", imgsz=512, conf=0.25)
```

### 3. 使用命令行推理

```bash
yolo predict model=weights/best.pt source=demo.jpg imgsz=512 conf=0.25
```

### 4. 导出模型

当前仓库已经包含 ONNX 文件，如需重新导出：

```bash
yolo export model=weights/best.pt format=onnx imgsz=512 simplify=True
```

## 结果可视化

## 可视化结果

![image](https://github.com/jikekei/dragon-fruit-stem-disease-yolov8/blob/81ecd22a6483523cac78bee7a64a33154e1e1ebb/results.png)
![image](https://github.com/jikekei/dragon-fruit-stem-disease-yolov8/blob/26614e1c84be6323c04da3ca358522de277d4ab8/BoxPR_curve.png)
![image](https://github.com/jikekei/dragon-fruit-stem-disease-yolov8/blob/26614e1c84be6323c04da3ca358522de277d4ab8/confusion_matrix.png)

当前目录中已包含：

- `results.png`：训练过程指标变化
- `BoxPR_curve.png`：Precision-Recall 曲线
- `BoxF1_curve.png`：F1 曲线
- `confusion_matrix.png`：混淆矩阵
- `val_batch*_pred.jpg`：验证集预测样例

## 适用场景

- 火龙果茎部病害自动识别
- 田间病害巡检与辅助诊断
- 智慧农业病害预警系统
- 农业视觉检测应用原型开发

## 已知限制

- 当前 README 基于训练结果整理，尚未补充数据集来源、样本量和采集条件
- 当前展示的是单次训练结果，尚未加入多组实验对比
- 若用于真实生产环境，建议继续验证复杂背景、遮挡和不同光照条件下的鲁棒性

## 数据集来源与采集说明

当前训练使用的数据集为 `datasets/dragon_stem_det`，是项目内整理后的标准 YOLO 检测数据集。

- 数据集名称：`dragon_stem_det`
- 任务类型：火龙果茎部病害目标检测
- 类别数量：5 类
- 数据集总量：545 张图像
- 划分方式：`train / val / test = 379 / 107 / 59`

根据项目文档，当前检测集由已有火龙果病害相关公开图像资料整理得到，并转换为适用于 YOLOv8 训练的检测格式。当前版本不包含 `Healthy` 健康类，发布的是 5 类病害检测集。

## 每一类病害的样本数量

| 类别 | train | val | test | 合计 |
| --- | ---: | ---: | ---: | ---: |
| Anthracnose | 72 | 20 | 11 | 103 |
| Brown_Stem_Spot | 81 | 23 | 13 | 117 |
| Gray_Blight | 81 | 23 | 12 | 116 |
| Soft_Rot | 77 | 22 | 12 | 111 |
| Stem_Canker | 68 | 19 | 11 | 98 |
| Total | 379 | 107 | 59 | 545 |

## 标注格式与标注规范

数据集采用标准 YOLO 目标检测标注格式，目录结构如下：

```text
datasets/dragon_stem_det/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── dataset.yaml
└── split_manifest.csv
```

标注文件与图像文件同名，对应关系示例：

- `images/train/Anthracnose_001.jpg`
- `labels/train/Anthracnose_001.txt`

YOLO 标签文件中每一行表示一个目标，格式为：

```text
class_id x_center y_center width height
```

其中：

- `class_id` 为类别编号，范围为 `0-4`
- 坐标采用相对图像宽高归一化后的数值
- 一个图像中有几个目标，标签文件就有几行

## 训练环境版本

当前可确认的训练/推理环境版本如下：

| 组件 | 版本 |
| --- | --- |
| Python | 3.14.0 |
| PyTorch | 2.10.0+cu126 |
| CUDA | 12.6 |
| Ultralytics | 8.4.21 |
| GPU | NVIDIA GeForce GTX 1060 3GB |

## 与其他检测模型或不同训练参数的对比结果

当前目录发布的是 `train_optimized` 这一版训练结果，主要配置为：

- 输入尺寸：`512`
- Epochs：`50`
- Batch Size：`8`
- Optimizer：`AdamW`
- Cosine LR：`true`
- 预训练微调：`true`

目前该 README 中尚未加入与 `YOLOv8n / YOLOv8s`、不同输入尺寸、不同优化器或不同增强策略的系统对比实验表。若后续补做对比实验，增加如下表格：

| 实验 | 模型/配置 | mAP@50 | mAP@50:95 | Precision | Recall | 备注 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Baseline | YOLOv8 baseline | - | - | - | - | 待补充 |
| Current | train_optimized | 0.6651 | 0.3972 | 0.6908 | 0.6410 | 当前最佳 epoch=45 |


## 致谢

本模型训练基于 Ultralytics YOLOv8 工具链完成，可作为火龙果病害识别任务的目标检测基线模型。
