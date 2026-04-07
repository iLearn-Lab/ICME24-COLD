import json
import argparse
from collections import defaultdict


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def normalize_keys(data):
    """移除视频文件名中的 .mp4 后缀，确保键名一致"""
    return {k.replace(".mp4", ""): v for k, v in data.items()}


def iouu(seg1, seg2):
    """
    用于合并逻辑的特定 IoU 计算
    Inter = sum of lengths, Union = coverage span
    """
    s1, e1 = seg1
    s2, e2 = seg2
    inter = e1 - s1 + e2 - s2
    union = max(e2, e1) - min(s1, s2)
    return inter / union if union > 0 else 0


def merge_segments(segments, merge_iou_threshold=0.5, merge_diff_keynum=True):
    """
    合并同一视频内的预测片段
    """
    if not segments:
        return []

    # 确保按开始时间排序
    segments = sorted(segments, key=lambda x: x["segment"][0])

    merged = []
    i = 0
    while i < len(segments):
        curr = segments[i].copy()
        j = i + 1
        while j < len(segments):
            next_seg = segments[j]
            # 标签不同则停止合并
            if curr["label"] != next_seg["label"]:
                break

            # 解析 key_num
            curr_key = int(curr["key_num"].replace('k', ''))
            next_key = int(next_seg["key_num"].replace('k', ''))

            curr_seg = curr["segment"]
            next_seg_coords = next_seg["segment"]

            if curr["key_num"] != next_seg["key_num"]:
                # 处理不同 key_num 的情况
                if merge_diff_keynum and next_key > curr_key:
                    curr["segment"][1] = next_seg_coords[1]
                    j += 1
                else:
                    break
            else:
                # 相同 key_num，检查 IoU
                iou_val = iouu(curr_seg, next_seg_coords)
                if iou_val > merge_iou_threshold:
                    curr["segment"][1] = next_seg_coords[1]
                    j += 1
                else:
                    break
        merged.append(curr)
        i = j
    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge overlapping prediction segments.")
    parser.add_argument("--input", type=str, required=True, help="Path to raw prediction JSON")
    parser.add_argument("--output", type=str, required=True, help="Path to save merged prediction JSON")
    parser.add_argument("--merge_iou", type=float, default=0.9, help="IoU threshold for merging the same key segments")
    parser.add_argument("--merge_diff_keynum", action="store_true", default=False,
                        help="Allow merging segments with different key_nums")

    args = parser.parse_args()

    print(f"Loading predictions from {args.input}...")
    pred_data = load_json(args.input)
    pred_data = normalize_keys(pred_data)

    print(f"Merging segments with iou_threshold={args.merge_iou}, diff_keynum={args.merge_diff_keynum}...")
    merged_count = 0
    total_count = 0

    for vid, info in pred_data.items():
        if "annotations" in info:
            original_len = len(info["annotations"])
            total_count += original_len
            info["annotations"] = merge_segments(
                info["annotations"],
                merge_iou_threshold=args.merge_iou,
                merge_diff_keynum=args.merge_diff_keynum
            )
            merged_count += (original_len - len(info["annotations"]))

    save_json(pred_data, args.output)
    print(f"Merging complete. Saved to {args.output}")
    print(f"Total segments: {total_count}, Removed: {merged_count}, Remaining: {total_count - merged_count}")


if __name__ == "__main__":
    main()