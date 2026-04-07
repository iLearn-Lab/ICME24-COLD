import torch
import cv2
import os
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor,Qwen2_5_VLForConditionalGeneration

# 指定模型和处理器的本地路径
model_path = "/root/.cache/modelscope/hub/Qwen/Qwen2___5-VL-7B-Instruct"
output_txt_file = "./output/caption.txt"
device = "cuda:1"

# 加载模型
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map=device,
    local_files_only=True  # 确保只从本地加载文件
)

# 加载处理器
processor = AutoProcessor.from_pretrained(model_path,use_fast=True)

def calculate_fps(video_path, target_frames=25):
    """
    动态计算 fps，以满足目标提取帧数量
    :param video_path: 视频路径
    :param target_frames: 目标提取帧数量
    :return: 动态计算的 fps
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频的总帧数
    cap.release()
    if total_frames == 0:
        raise ValueError("Failed to get total frames from the video.")
    fps = 30 * target_frames / total_frames
    return fps

def generate_caption_for_video(video_name, category, video_path):
    """
    根据视频名称和类别生成视频的caption
    :param video_name: 视频文件名
    :param category: 动作类别
    :param video_path: 视频路径
    :return: 视频的生成caption
    """
    dynamic_fps = calculate_fps(video_path)

    # 定义消息，其中包括视频和文本内容
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "fps": dynamic_fps,
                },
                {
                    "type": "text",
                    "text": (
                        f"The video contains multiple athletic actions. "
                        f"Please focus on identifying and describing the {category} action only. "
                        f"Provide a detailed description of the {category} action, "
                        "including the people's stance, motion, and other relevant details specific to this action. "
                        "Do not generate duplicate text descriptions."
                    ),
                },
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # 增加 max_new_tokens 参数以适应较长输出
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]

def main():
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_txt_file), exist_ok=True)

    # 读取视频类别文件
    input_txt_file = "./output/category.txt"
    with open(input_txt_file, "r") as file:
        video_category_lines = file.readlines()

    # 生成视频的描述并保存
    with open(output_txt_file, "w") as txt_file:
        for line in video_category_lines:
            video_name, category = line.strip().split(" - ")
            video_path = os.path.join("./activitynet/videos/v1-3/val", video_name)
            try:
                caption = generate_caption_for_video(video_name, category, video_path)
                txt_file.write(f"{video_name}*-{category}*-{caption}\n")
                print(f"{video_name} - yes")
            except Exception as e:
                print(f"{video_name} - no: {e}")

    print(f"all have to {output_txt_file}")

if __name__ == "__main__":
    main()
