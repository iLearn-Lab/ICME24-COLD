import json
import os
from collections import defaultdict
from openai import OpenAI

# 配置参数
API_KEY = ""
BASE_URL = "https://api.deepseek.com"
INPUT_FILE = "./output/caption.txt"
OUTPUT_FILE = "./output/stage.json"

# 初始化 DeepSeek 客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def generate_action_stages(action, caption):
    #use DeepSeek to divide the action into stage with key flag
    prompt = f"""Analyze the action and break it into distinct stages. Follow these rules:
                1. Action: {action}
                2. Context: {caption}
                3. Decompose into 2-5 essential stages
                4. Each stage should be the people do sth.
                5. Avoid overlapping stages
                6. Maintain chronological order
                7. Mark stages unique to this action with * (only if the stage is distinctive and unlikely to appear in other actions)
                Output format:
                Long jump:People are running up,people jump*,People jumping in the air*,People landing
                ONLY output the text in that format without any extra content."""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": "You are a helpful assistant"},
                      {"role": "user", "content": prompt}],
            temperature=0.5,
            stream=False
        )
        generated_text = response.choices[0].message.content.strip()
        generated_text = generated_text.replace('\n', ' ').replace(' ,', ',').strip()

        # Analyze the returned text
        if ':' not in generated_text:
            raise ValueError("lack the :::")
        _, stages_part = generated_text.split(':', 1)
        stages_list = [stage.strip() for stage in stages_part.split(',') if stage.strip()]
        return {action: {"stages": stages_list}}
    except Exception as e:
        print(f"caption error: {str(e)}")
        return {action: {"stages": []}}


def load_existing_output():
    """加载已存在的输出文件"""
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            try:
                return json.load(f)
            except:
                return {}
    return {}


def save_partial_output(output_data):
    """保存当前输出到文件"""
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def process_input_file():
    """逐行处理输入文件，支持断点续跑"""
    output_data = load_existing_output()
    with open(INPUT_FILE, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                line = line.strip()
                if '.mp4*-' not in line:
                    raise ValueError("video don't .mp4*- ")
                video_part, remaining = line.split('.mp4*-', 1)
                video_name = video_part + '.mp4'

                # 如果该视频已处理，跳过
                if video_name in output_data:
                    print(f"skip: {video_name}")
                    continue

                if '*-' not in remaining:
                    raise ValueError("lack caption or *- ")
                action, caption = remaining.split('*-', 1)
                action, caption = action.strip(), caption.strip()

                result = generate_action_stages(action, caption)
                if action in result:
                    output_data[video_name] = {action: result[action]}
                    save_partial_output(output_data)  # 每处理一个就保存
                    print(f"process: {video_name} - {action}")
                else:
                    print(f"lack: {video_name}:{action}")
            except Exception as e:
                print(f"error @ line {line_num}: {str(e)} - context: {line}")

    return output_data


if __name__ == "__main__":
    final_output = process_input_file()
    print(f"sucess: {OUTPUT_FILE}")
    print(f"sum: {len(final_output)}")
    print(f"sum action: {sum(len(v) for v in final_output.values())}")
