#!/usr/bin/env python3
"""
BPH 失败病例可视化复核脚本

功能：
1. 从 summary.json 中筛选 BPH 失败病例
2. 生成多模态影像 (T2, ADC, DWI) + 金标准 + 预测的叠加可视化
3. 支持三种视图：轴位 (axial)、矢状位 (sagittal)、冠状位 (coronal)
4. 自动定位病灶区域，生成关键切片

用法：
    python script/visualize_bph_failures.py --case BPH_0000044758 --output viz/failures
    python script/visualize_bph_failures.py --all-failures --output viz/failures
    python script/visualize_bph_failures.py --top-n 10 --output viz/failures
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

warnings.filterwarnings('ignore')


# 项目路径配置
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # 从 v1 目录向上到项目根目录
SUMMARY_JSON = PROJECT_ROOT / "nnUNet_results/Dataset101_Prostate/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation/summary.json"
IMAGES_DIR = PROJECT_ROOT / "nnUNet_raw/Dataset101_Prostate/imagesTr"
LABELS_DIR = PROJECT_ROOT / "nnUNet_raw/Dataset101_Prostate/labelsTr"
PREDICTIONS_DIR = PROJECT_ROOT / "nnUNet_results/Dataset101_Prostate/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation"

# 模态映射 (nnUNet 命名规范)
MODALITY_MAP = {
    0: "T2_not_fs",
    1: "ADC", 
    2: "DWI"
}


def load_summary() -> Dict:
    """加载验证结果 summary.json"""
    with open(SUMMARY_JSON, 'r') as f:
        return json.load(f)


def get_bph_failures(summary: Dict, mode: str = "all", top_n: int = None) -> List[Dict]:
    """
    筛选 BPH 失败病例
    
    Args:
        summary: summary.json 内容
        mode: 'all' (所有BPH), 'zero_dice' (Dice=0), 'false_positive' (Dice=0但n_pred>0)
        top_n: 只取前N个
    """
    failures = []
    
    for case_data in summary.get("metric_per_case", []):
        pred_file = case_data["prediction_file"]
        case_name = Path(pred_file).stem
        
        # 只保留 BPH 病例
        if not case_name.startswith("BPH_"):
            continue
            
        metrics = case_data["metrics"]["1"]  # foreground class
        dice = metrics["Dice"]
        n_pred = metrics["n_pred"]
        n_ref = metrics["n_ref"]
        
        # 根据模式筛选
        if mode == "zero_dice" and dice > 0:
            continue
        elif mode == "false_positive" and not (dice == 0 and n_pred > 0):
            continue
        elif mode == "complete_miss" and not (dice == 0 and n_pred == 0):
            continue
            
        failures.append({
            "case_name": case_name,
            "dice": dice,
            "n_pred": n_pred,
            "n_ref": n_ref,
            "tp": metrics["TP"],
            "fp": metrics["FP"],
            "fn": metrics["FN"],
            "prediction_file": pred_file,
            "reference_file": case_data["reference_file"]
        })
    
    # 按 n_ref (病灶大小) 降序排列，优先看大病灶失败
    failures.sort(key=lambda x: x["n_ref"], reverse=True)
    
    if top_n:
        failures = failures[:top_n]
        
    return failures


def load_case_data(case_name: str) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray, nib.Nifti1Image]:
    """
    加载病例的影像、标签和预测
    
    Returns:
        images: {modality_idx: data_array}
        label: ground truth array
        prediction: prediction array
        affine: nifti affine matrix
    """
    # 加载各模态影像
    images = {}
    affine = None
    for mod_idx, mod_name in MODALITY_MAP.items():
        img_path = IMAGES_DIR / f"{case_name}_{mod_idx:04d}.nii.gz"
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        nii = nib.load(str(img_path))
        images[mod_idx] = np.asarray(nii.dataobj)
        if affine is None:
            affine = nii
    
    # 加载金标准
    label_path = LABELS_DIR / f"{case_name}.nii.gz"
    if not label_path.exists():
        raise FileNotFoundError(f"Label not found: {label_path}")
    label = np.asarray(nib.load(str(label_path)).dataobj)
    
    # 加载预测
    pred_path = PREDICTIONS_DIR / f"{case_name}.nii.gz"
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction not found: {pred_path}")
    prediction = np.asarray(nib.load(str(pred_path)).dataobj)
    
    return images, label, prediction, affine


def find_roi_slices(label: np.ndarray, prediction: np.ndarray, 
                    margin: int = 5) -> Dict[str, List[int]]:
    """
    找到包含病灶的关键切片位置
    
    Returns:
        {view_name: [slice_indices]}
    """
    # 合并金标准和预测，找到所有相关区域
    combined = ((label > 0) | (prediction > 0)).astype(np.uint8)
    
    if combined.sum() == 0:
        # 如果都没有，返回中间切片
        z_mid = combined.shape[2] // 2
        return {
            "axial": [z_mid],
            "sagittal": [combined.shape[0] // 2],
            "coronal": [combined.shape[1] // 2]
        }
    
    # 找到非零区域的边界
    nonzero = np.argwhere(combined > 0)
    z_min, z_max = nonzero[:, 2].min(), nonzero[:, 2].max()
    x_min, x_max = nonzero[:, 0].min(), nonzero[:, 0].max()
    y_min, y_max = nonzero[:, 1].min(), nonzero[:, 1].max()
    
    # 选择关键切片（边界、中间）
    def select_slices(min_idx, max_idx, n=3):
        """选择 n 个代表性切片"""
        if max_idx - min_idx < n:
            return list(range(min_idx, max_idx + 1))
        step = (max_idx - min_idx) // (n - 1)
        return [min_idx + i * step for i in range(n)]
    
    return {
        "axial": select_slices(max(0, z_min - margin), min(combined.shape[2] - 1, z_max + margin)),
        "sagittal": select_slices(max(0, x_min - margin), min(combined.shape[0] - 1, x_max + margin)),
        "coronal": select_slices(max(0, y_min - margin), min(combined.shape[1] - 1, y_max + margin))
    }


def normalize_image(image: np.ndarray, modality: int) -> np.ndarray:
    """归一化影像到 0-1 范围"""
    # 使用百分位数裁剪极端值
    p1, p99 = np.percentile(image[image > 0], [1, 99]) if (image > 0).any() else (0, 1)
    if p99 > p1:
        image = np.clip(image, p1, p99)
        image = (image - p1) / (p99 - p1)
    else:
        image = np.zeros_like(image)
    return np.clip(image, 0, 1)


def create_overlay_figure(images: Dict[int, np.ndarray], 
                          label: np.ndarray, 
                          prediction: np.ndarray,
                          case_info: Dict,
                          view: str = "axial",
                          slice_idx: int = 0) -> plt.Figure:
    """
    创建叠加可视化图
    
    布局：
    - 第一行：T2, ADC, DWI 原图
    - 第二行：T2+GT(红), T2+Pred(绿), T2+GT+Pred(叠加)
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"{case_info['case_name']} | {view.upper()} Slice {slice_idx} | "
                 f"Dice={case_info['dice']:.4f} | n_ref={case_info['n_ref']:.0f} | n_pred={case_info['n_pred']:.0f}",
                 fontsize=12, fontweight='bold')
    
    # 根据视图选择切片
    if view == "axial":
        t2_slice = images[0][:, :, slice_idx]
        adc_slice = images[1][:, :, slice_idx]
        dwi_slice = images[2][:, :, slice_idx]
        label_slice = label[:, :, slice_idx]
        pred_slice = prediction[:, :, slice_idx]
    elif view == "sagittal":
        t2_slice = images[0][slice_idx, :, :]
        adc_slice = images[1][slice_idx, :, :]
        dwi_slice = images[2][slice_idx, :, :]
        label_slice = label[slice_idx, :, :]
        pred_slice = prediction[slice_idx, :, :]
    else:  # coronal
        t2_slice = images[0][:, slice_idx, :]
        adc_slice = images[1][:, slice_idx, :]
        dwi_slice = images[2][:, slice_idx, :]
        label_slice = label[:, slice_idx, :]
        pred_slice = prediction[:, slice_idx, :]
    
    # 归一化
    t2_norm = normalize_image(t2_slice, 0)
    
    # 第一行：原图
    axes[0, 0].imshow(t2_norm, cmap='gray', aspect='auto')
    axes[0, 0].set_title('T2 (not fs)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(normalize_image(adc_slice, 1), cmap='gray', aspect='auto')
    axes[0, 1].set_title('ADC')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(normalize_image(dwi_slice, 2), cmap='gray', aspect='auto')
    axes[0, 2].set_title('DWI')
    axes[0, 2].axis('off')
    
    # 第二行：叠加图
    # T2 + Ground Truth (红色)
    axes[1, 0].imshow(t2_norm, cmap='gray', aspect='auto')
    gt_overlay = np.zeros((*label_slice.shape, 4))  # RGBA
    gt_overlay[label_slice > 0] = [1, 0, 0, 0.5]  # 红色半透明
    axes[1, 0].imshow(gt_overlay, aspect='auto')
    axes[1, 0].set_title('T2 + GT (Red)')
    axes[1, 0].axis('off')
    
    # T2 + Prediction (绿色)
    axes[1, 1].imshow(t2_norm, cmap='gray', aspect='auto')
    pred_overlay = np.zeros((*pred_slice.shape, 4))
    pred_overlay[pred_slice > 0] = [0, 1, 0, 0.5]  # 绿色半透明
    axes[1, 1].imshow(pred_overlay, aspect='auto')
    axes[1, 1].set_title('T2 + Pred (Green)')
    axes[1, 1].axis('off')
    
    # T2 + GT + Prediction (叠加)
    axes[1, 2].imshow(t2_norm, cmap='gray', aspect='auto')
    combined_overlay = np.zeros((*label_slice.shape, 4))
    # GT = 红色, Pred = 绿色, 重叠 = 黄色
    combined_overlay[(label_slice > 0) & (pred_slice == 0)] = [1, 0, 0, 0.5]  # FN: 红色
    combined_overlay[(label_slice == 0) & (pred_slice > 0)] = [0, 1, 0, 0.5]  # FP: 绿色
    combined_overlay[(label_slice > 0) & (pred_slice > 0)] = [1, 1, 0, 0.5]   # TP: 黄色
    axes[1, 2].imshow(combined_overlay, aspect='auto')
    axes[1, 2].set_title('Overlay: FN(Red) FP(Green) TP(Yellow)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    return fig


def visualize_case(case_info: Dict, output_dir: Path, views: List[str] = None):
    """可视化单个病例"""
    if views is None:
        views = ["axial"]  # 默认只生成轴位
    
    case_name = case_info["case_name"]
    print(f"Processing {case_name}...")
    
    try:
        # 加载数据
        images, label, prediction, affine = load_case_data(case_name)
        
        # 找到关键切片
        roi_slices = find_roi_slices(label, prediction)
        
        # 为每个视图生成图像
        for view in views:
            slices = roi_slices.get(view, [0])
            for slice_idx in slices:
                fig = create_overlay_figure(images, label, prediction, case_info, view, slice_idx)
                
                # 保存
                output_path = output_dir / f"{case_name}_{view}_slice{slice_idx:03d}.png"
                fig.savefig(output_path, dpi=150, bbox_inches='tight', 
                           facecolor='black', edgecolor='none')
                plt.close(fig)
        
        print(f"  ✓ Saved to {output_dir}/{case_name}_*.png")
        
    except Exception as e:
        print(f"  ✗ Error processing {case_name}: {e}")


def generate_comparison_report(failures: List[Dict], output_dir: Path):
    """生成对比报告表格"""
    report_path = output_dir / "bph_failures_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("BPH 失败病例复核清单\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"{'Case Name':<20} {'Dice':>8} {'n_ref':>10} {'n_pred':>10} "
                f"{'TP':>8} {'FP':>8} {'FN':>8} {'Recall':>8} {'Precision':>10}\n")
        f.write("-" * 100 + "\n")
        
        for case in failures:
            recall = case['tp'] / (case['tp'] + case['fn']) if (case['tp'] + case['fn']) > 0 else 0
            precision = case['tp'] / (case['tp'] + case['fp']) if (case['tp'] + case['fp']) > 0 else 0
            
            f.write(f"{case['case_name']:<20} {case['dice']:>8.4f} {case['n_ref']:>10.0f} "
                    f"{case['n_pred']:>10.0f} {case['tp']:>8.0f} {case['fp']:>8.0f} "
                    f"{case['fn']:>8.0f} {recall:>8.3f} {precision:>10.3f}\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("失败模式分类:\n")
        f.write("-" * 100 + "\n")
        
        complete_miss = [c for c in failures if c['dice'] == 0 and c['n_pred'] == 0]
        false_pos_only = [c for c in failures if c['dice'] == 0 and c['n_pred'] > 0]
        low_dice = [c for c in failures if 0 < c['dice'] < 0.3]
        
        f.write(f"完全漏检 (Dice=0, n_pred=0): {len(complete_miss)} 例\n")
        for c in complete_miss[:5]:  # 只列前5个
            f.write(f"  - {c['case_name']}: n_ref={c['n_ref']:.0f}\n")
        
        f.write(f"\n零交并+误检 (Dice=0, n_pred>0): {len(false_pos_only)} 例\n")
        for c in false_pos_only[:5]:
            f.write(f"  - {c['case_name']}: n_ref={c['n_ref']:.0f}, n_pred={c['n_pred']:.0f}\n")
        
        f.write(f"\n低Dice (0<Dice<0.3): {len(low_dice)} 例\n")
        for c in low_dice[:5]:
            f.write(f"  - {c['case_name']}: Dice={c['dice']:.4f}\n")
    
    print(f"\nReport saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="BPH 失败病例可视化复核")
    parser.add_argument("--case", type=str, help="指定单个病例，如 BPH_0000044758")
    parser.add_argument("--all-failures", action="store_true", help="可视化所有 BPH 失败病例")
    parser.add_argument("--mode", type=str, default="zero_dice", 
                       choices=["all", "zero_dice", "false_positive", "complete_miss"],
                       help="失败病例筛选模式")
    parser.add_argument("--top-n", type=int, help="只处理前N个病例（按病灶大小排序）")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "viz" / "bph_failures",
                       help="输出目录")
    parser.add_argument("--views", nargs="+", default=["axial"],
                       choices=["axial", "sagittal", "coronal"],
                       help="要生成的视图")
    parser.add_argument("--report-only", action="store_true", help="只生成报告，不生成图像")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载 summary
    print(f"Loading summary from {SUMMARY_JSON}")
    summary = load_summary()
    
    # 确定要处理的病例
    if args.case:
        # 单个病例
        case_info = None
        for case_data in summary.get("metric_per_case", []):
            pred_file = case_data["prediction_file"]
            case_name = Path(pred_file).stem
            if case_name == args.case:
                metrics = case_data["metrics"]["1"]
                case_info = {
                    "case_name": case_name,
                    "dice": metrics["Dice"],
                    "n_pred": metrics["n_pred"],
                    "n_ref": metrics["n_ref"],
                    "tp": metrics["TP"],
                    "fp": metrics["FP"],
                    "fn": metrics["FN"],
                }
                break
        
        if case_info is None:
            print(f"Error: Case {args.case} not found in summary")
            return
        
        failures = [case_info]
    else:
        # 批量处理
        failures = get_bph_failures(summary, mode=args.mode, top_n=args.top_n)
        print(f"Found {len(failures)} BPH failure cases (mode={args.mode})")
    
    # 生成报告
    generate_comparison_report(failures, output_dir)
    
    if args.report_only:
        print("Report-only mode, skipping image generation")
        return
    
    # 生成可视化
    print(f"\nGenerating visualizations...")
    print(f"Output directory: {output_dir}")
    print(f"Views: {args.views}")
    print("-" * 60)
    
    for case_info in failures:
        visualize_case(case_info, output_dir, args.views)
    
    print("\n" + "=" * 60)
    print(f"Visualization complete! Check: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
