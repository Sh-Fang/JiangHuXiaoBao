import whisper
import time
import json
import os

# 加载 Whisper 模型
load_time = time.time()
model = whisper.load_model("medium").to("cuda")
print(f"Model loaded in {time.time() - load_time:.2f}s")

# 定义片段路径
segment_paths = [
    "output/segment_0/segment_0/vocals.wav",
    "output/segment_10/segment_10/vocals.wav",
    "output/segment_20/segment_20/vocals.wav",
    "output/segment_30/segment_30/vocals.wav",
    "output/segment_40/segment_40/vocals.wav",
]

# 处理每个片段
for segment_path in segment_paths:
    # 检查文件是否存在
    if not os.path.exists(segment_path):
        print(f"File not found: {segment_path}")
        continue

    # 处理音频
    process_time = time.time()
    result = model.transcribe(segment_path, fp16=True, language="zh")
    print(f"Transcription completed for {segment_path} in {time.time() - process_time:.2f}s")

    # 生成 JSON 文件名（基于片段编号）
    segment_number = segment_path.split("/")[1]  # 提取片段编号，如 "segment_0"
    output_file = f"output/{segment_number}/transcription.json"

    # 保存为 JSON 文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result["text"], f, ensure_ascii=False, indent=4)

    print(f"Transcription saved to {output_file}")

print("All transcriptions completed.")
