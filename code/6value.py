import json
import numpy as np
import argparse
import random
from collections import defaultdict


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def normalize_keys(data):
    return {k.replace(".mp4", ""): v for k, v in data.items()}


def iou(seg1, seg2):
    """标准 IoU 计算，用于评估"""
    s1, e1 = seg1
    s2, e2 = seg2
    inter = max(0, min(e1, e2) - max(s1, s2))
    union = max(e2, e1) - min(s1, s2)
    return inter / union if union > 0 else 0


def compute_ap(gt_segments, pred_segments, iou_threshold):
    """计算单个类别的 Average Precision"""
    # 按置信度排序，如果没有 score 默认为 1.0
    pred_segments = sorted(pred_segments, key=lambda x: x.get("score", 1.0), reverse=True)
    gt_used = [False] * len(gt_segments)
    tp = np.zeros(len(pred_segments))
    fp = np.zeros(len(pred_segments))

    for i, pred in enumerate(pred_segments):
        found = False
        for j, gt in enumerate(gt_segments):
            if gt["label"] != pred["label"]:
                continue
            if iou(pred["segment"], gt["segment"]) >= iou_threshold:
                if not gt_used[j]:
                    tp[i] = 1
                    gt_used[j] = True
                    found = True
                    break
        if not found:
            fp[i] = 1

    if tp.sum() + fp.sum() == 0:
        return 0.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precision = tp_cum / (tp_cum + fp_cum + 1e-6)
    recall = tp_cum / (len(gt_segments) + 1e-6)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return ap


def evaluate_map(gt_data, pred_data, iou_thresholds, label_filter=None):
    """
    计算 mAP。注意：此处假设 pred_data 已经过 merge_segments 处理。
    """
    common_keys = set(gt_data.keys()) & set(pred_data.keys())
    results = {f"mAP@{iou:.2f}": [] for iou in iou_thresholds}

    for vid in common_keys:
        gt_ann = [{"segment": ann["segment"], "label": ann["label"]} for ann in gt_data[vid]["annotations"]]
        # 预测数据假设已经合并过，直接使用
        pred_ann = pred_data[vid]["annotations"]

        if label_filter is not None:
            gt_ann = [ann for ann in gt_ann if ann["label"] in label_filter]
            pred_ann = [ann for ann in pred_ann if ann["label"] in label_filter]

        if len(gt_ann) == 0:
            continue

        for iou_t in iou_thresholds:
            ap = compute_ap(gt_ann, pred_ann, iou_t)
            results[f"mAP@{iou_t:.2f}"].append(ap)

    mean_results = {}
    for k, v in results.items():
        mean_results[k] = np.mean(v) if v else 0.0

    # 计算平均 mAP (avg_mAP)
    mAP_values = [v for k, v in mean_results.items() if k.startswith("mAP@")]
    mean_results["avg_mAP"] = np.mean(mAP_values) if mAP_values else 0.0
    return mean_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate mAP with Seen/Unseen split.")
    parser.add_argument("--gt", type=str, required=True, help="Path to Ground Truth JSON")
    parser.add_argument("--pred", type=str, required=True, help="Path to Merged Prediction JSON")
    parser.add_argument("--label_threshold", type=int, default=0, help="Minimum video count for a label to be included")
    parser.add_argument("--seeds", type=int, nargs='+', default=[1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999],
                        help="Random seeds for splitting")
    parser.add_argument("--iou_steps", type=float, default=0.05,
                        help="Step size for IoU thresholds (e.g., 0.05 for 0.5, 0.55...)")

    args = parser.parse_args()

    # 生成 IoU 阈值列表 [0.5, 0.55, ..., 0.95]
    iou_thresholds = np.arange(0.5, 1.0, args.iou_steps)

    print(f"Loading GT from {args.gt}...")
    gt_data = normalize_keys(load_json(args.gt))
    print(f"Loading Merged Pred from {args.pred}...")
    pred_data = normalize_keys(load_json(args.pred))

    # 统计 label 出现频次（从 ground truth 获取）
    label_to_videos = defaultdict(set)
    for vid, info in gt_data.items():
        for ann in info["annotations"]:
            label_to_videos[ann["label"]].add(vid)

    label_video_count = {label: len(vset) for label, vset in label_to_videos.items()}
    filtered_labels = [label for label, count in label_video_count.items() if count > args.label_threshold]

    print(f"Total valid labels: {len(filtered_labels)}")
    print(f"Running evaluation with {len(args.seeds)} seeds...")

    seen_maps = defaultdict(list)
    unseen_maps = defaultdict(list)

    for seed in args.seeds:
        random.seed(seed)
        labels = filtered_labels.copy()
        random.shuffle(labels)
        split = int(len(labels) * 0.75)
        seen_labels = labels[:split]
        unseen_labels = labels[split:]

        seen_result = evaluate_map(gt_data, pred_data, iou_thresholds, label_filter=seen_labels)
        unseen_result = evaluate_map(gt_data, pred_data, iou_thresholds, label_filter=unseen_labels)

        for k in seen_result:
            seen_maps[k].append(seen_result[k])
            unseen_maps[k].append(unseen_result[k])

    print("\n====== Final Averaged Results ======")
    print("---- Unseen Results (Generalization) ----")
    for k in iou_thresholds:
        key = f"mAP@{k:.2f}"
        # 取消下面注释可打印每个阈值
        # print(f"{key}: {np.mean(unseen_maps[key]):.4f}")
    print(f"avg_mAP: {np.mean(unseen_maps['avg_mAP']):.3f}")

    # 如果需要查看 Seen 结果，可取消下面注释
    # print("\n---- Seen Results ----")
    # print(f"avg_mAP: {np.mean(seen_maps['avg_mAP']):.3f}")


if __name__ == "__main__":
    main()