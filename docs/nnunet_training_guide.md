# nnU-Net v2 前列腺多模态分割训练操作文档

## 项目概述

本文档记录了使用 nnU-Net v2 的一键式 Baseline 训练流程，用于前列腺癌多模态 MRI 分割（BPH vs PCA）。

- **数据集**: Dataset101_Prostate  
- **输入模态**: T2_not_fs（通道0）、ADC（通道1）、DWI（通道2）  
- **分割目标**: 前列腺 / 前列腺病灶区域（二分类：背景 0，前景 1）  
- **训练案例数**: 388 例（BPH: ~197，PCA: ~191，类别均衡）  

---

## 目录结构说明

```
nnunet/
├── FilteredData/               ← 原始筛选后的纯净数据（未修改）
│   ├── BPH/{case_id}/         ← 含 ADC.nii, DWI.nii, T2_not_fs.nii, ROI.nii
│   └── PCA/{case_id}/
├── nnUNet_raw/                 ← nnU-Net 标准格式数据（已转换）
│   └── Dataset101_Prostate/
│       ├── dataset.json        ← 数据集描述文件
│       ├── imagesTr/           ← 训练输入图像（_0000=T2, _0001=ADC, _0002=DWI）
│       └── labelsTr/           ← 训练标注（已二值化清洗，仅含 0/1）
├── nnUNet_preprocessed/        ← nnU-Net 预处理输出（自动生成）
├── nnUNet_results/             ← 训练结果和模型权重（自动生成）
├── convert_to_nnunet_format.py ← 数据格式转换脚本
└── docs/
    └── nnunet_training_guide.md ← 本文档
```

---

## 数据质量说明

| 检查项 | 结论 |
|--------|------|
| 文件完整性 | ✅ 全部 388 例的 3 个模态 + ROI 均完整存在 |
| 各模态 Shape 对齐 | ✅ 所有模态与 ROI 空间完全一致（筛选时强制要求） |
| 全零图像 | ✅ 已全部过滤 |
| ROI 二值化 | ✅ 脏污的连续浮点标签已在转换时全部修复（`data > 0.5`） |
| 类别均衡 | ✅ BPH 50.8% / PCA 49.2%，无需重采样 |

---

## ⚠️ 临床语义与训练策略核心提示（必读）

根据对现有数据 ROI 的体积统计，**当前数据集中的标注极大概率为前列腺病灶/结节 (Lesion)，而非整个前列腺腺体 (Whole Prostate)。** 中位数病灶体积仅占全视野的约 0.5%（数百科体素），最小的病灶甚至只有 18 个体素。

基于此重大发现，针对现有的 Baseline 训练方案做出以下声明备忘：

1. **当前策略（单类病灶分割）**：
   - 现有的转换数据**将 BPH（良性增生结节）和 PCA（癌变区域）混为一谈，全部看作 `Target=1` 进行训练**。
   - **适用场景**：这是最鲁棒的 Baseline。能够训练出一个"寻找前列腺内任何异常结节/病灶"的检测模型。由于目标极小，nnU-Net 将自动处理极度的类别不平衡和 Oversampling 工作，现有的配置已经是最佳起点。
   
2. **进阶策略探讨（向医生确认后升级）**：
   - **多类分割区分良恶性**：如果你需要模型直接判定是癌灶还是良性增生，强烈建议在将来的迭代中，修改标签：将 BPH 的 ROI 赋值为 1，PCA 的 ROI 赋值为 2，然后修改 `dataset.json` 进行多类分割。
   - **级联网络 (Cascade)**：如果小病灶分割准确率遇到瓶颈，可考虑引入外部的前列腺模型先切出前列腺腺体（Stage 1），再在内部切出微小病灶（Stage 2）。

---

## Step 1：安装 nnU-Net v2

```bash
pip install nnunetv2
```

如已安装，确认版本：
```bash
python -c "import nnunetv2; print(nnunetv2.__version__)"
```

---

## Step 2：设置环境变量

> ⚠️ **每次新开终端都需要重新设置**，建议写入系统环境变量或激活脚本。

**PowerShell：**
```powershell
$env:nnUNet_raw      = "d:\code\PycharmProjects\Prostate-Cancer-Multimodal-Segmentation\DataAnalysis\nnunet\nnUNet_raw"
$env:nnUNet_preprocessed = "d:\code\PycharmProjects\Prostate-Cancer-Multimodal-Segmentation\DataAnalysis\nnunet\nnUNet_preprocessed"
$env:nnUNet_results  = "d:\code\PycharmProjects\Prostate-Cancer-Multimodal-Segmentation\DataAnalysis\nnunet\nnUNet_results"
```

**CMD：**
```cmd
set nnUNet_raw=d:\code\PycharmProjects\Prostate-Cancer-Multimodal-Segmentation\DataAnalysis\nnunet\nnUNet_raw
set nnUNet_preprocessed=d:\code\PycharmProjects\Prostate-Cancer-Multimodal-Segmentation\DataAnalysis\nnunet\nnUNet_preprocessed
set nnUNet_results=d:\code\PycharmProjects\Prostate-Cancer-Multimodal-Segmentation\DataAnalysis\nnunet\nnUNet_results
```

---

## Step 3：数据集校验 + 自动规划预处理

nnU-Net 会自动分析数据集的几何特征（Shape、Spacing）并计算最优的：
- 目标重采样间距（Target Spacing）  
- 输入 Patch Size  
- 网络深度和 Feature Map 数量  
- 归一化策略（对 MRI 自动使用 `zscore`）  

```bash
nnUNetv2_plan_and_preprocess -d 101 --verify_dataset_integrity
```

> 💡 参数 `--verify_dataset_integrity` 会自动检查文件命名和格式规范，建议首次运行必加。  
> 预计耗时：5~15 分钟（取决于样本量和磁盘速度）。

---

## Step 4：启动训练

nnU-Net 默认使用 **5折交叉验证（5-Fold Cross-Validation）**。  
每次运行训练一个 fold，`FOLD` 取值为 `0, 1, 2, 3, 4`。

**训练 3D Full Resolution（推荐首选）：**
```bash
nnUNetv2_train 101 3d_fullres 0
nnUNetv2_train 101 3d_fullres 1
nnUNetv2_train 101 3d_fullres 2
nnUNetv2_train 101 3d_fullres 3
nnUNetv2_train 101 3d_fullres 4
```

> 💡 如想训练全部 5 折并生成最终集成模型，需要全部跑完。  
> 如只想快速验证效果，只跑 `fold 0` 即可得到一份可用的模型。

**可选：加 `--npz` 参数保留每个 epoch 的预测结果（方便 ensemble）：**
```bash
nnUNetv2_train 101 3d_fullres 0 --npz
```

**训练期间监控指标：**  
nnU-Net 会在终端实时打印每个 epoch 的 `train_loss`、`val_loss`、`Dice (pseudo)`。  
完整训练日志保存在：
```
nnUNet_results/Dataset101_Prostate/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/training_log_*.txt
```

---

## Step 5：查看训练结果与预测

**汇总 5 折验证结果：**
```bash
nnUNetv2_find_best_configuration 101 -c 3d_fullres
```

**对新的测试数据进行推理：**
```bash
nnUNetv2_predict -i <输入图像文件夹> -o <输出文件夹> -d 101 -c 3d_fullres -f all
```

> 输入测试图像须遵循 nnU-Net 命名格式：  
> `{case_id}_0000.nii.gz`（T2）、`{case_id}_0001.nii.gz`（ADC）、`{case_id}_0002.nii.gz`（DWI）

---

## 预期性能参考

由于我们确认了任务性质是**微小癌灶/病灶分割**，而非简单的器官分割，评估分数的期望阈值应大幅下调修正：

| 指标 | 目标对象 | 参考区间（业界公认标准） |
|------|---------|--------------------------|
| Dice Score | 整个前列腺腺体 | 0.85 ~ 0.92（非常容易）|
| **Dice Score** | **前列腺癌灶/结节 (Lesion)** | **0.60 ~ 0.75（难度极高）** |
| 95% Hausdorff Distance | 前列腺癌灶/结节 | 约 2 ~ 8 mm |
| 训练时间 | 单 fold，按本数据集大小 (RTX 3090) | 约 8~16 小时 |

> **提示**：如果病灶过小（例如只有十几个像素），Dice Score 由于对单像素错误极为敏感，容易发生断崖式下跌。在评估模型时，可以结合**病灶检测率（Detection Rate / Sensitivity）** 一并参考，不要单纯迷信 Dice 分数。

---

## 常见问题（FAQ）

**Q: 提示 `RuntimeError: CUDA out of memory`**  
A: 减小 batch size。在 nnU-Net 中可通过覆盖 `nnUNetPlans.json` 中的 `batch_size` 字段实现：
```json
"batch_size": 1
```

**Q: `verify_dataset_integrity` 报 channel 数量不对**  
A: 检查 `imagesTr` 里是否每个 case 都有 `_0000`, `_0001`, `_0002` 三个文件，且 `dataset.json` 的 `channel_names` 共 3 个。

**Q: 想只用 T2 单模态跑 ablation study**  
A: 重新转换数据只输出 `_0000.nii.gz`（仅 T2），并修改 `dataset.json` 中 `channel_names` 只保留 1 个通道，然后用新的 Dataset id（如 `102`）重新运行即可。

---

## 数据转换脚本说明

如需重新转换数据（修改模态组合等），运行：

```bash
cd d:\code\PycharmProjects\Prostate-Cancer-Multimodal-Segmentation\DataAnalysis\nnunet
python convert_to_nnunet_format.py
```

脚本关键逻辑：
- **输入**：`FilteredData/{BPH,PCA}/{case_id}/` 下的各 `.nii` 文件  
- **ROI 清洗**：`(roi > 0.5).astype(np.uint8)` 将所有插值污染值二值化  
- **格式转换**：图像保存为 `float32`，标注保存为 `uint8`，统一压缩为 `.nii.gz`  
- **输出**：`nnUNet_raw/Dataset101_Prostate/imagesTr` 和 `labelsTr`  
