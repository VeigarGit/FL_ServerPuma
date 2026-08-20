import torch
from transformers import AutoProcessor
model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
try:
    processor = AutoProcessor.from_pretrained(model_id)
    print("Processor loaded for 2.5-VL")
except Exception as e:
    print("Failed 2.5-VL:", e)
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    print("Processor loaded for 2-VL")

from PIL import Image
import numpy as np
image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
messages = [
    {"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Classify this image."}
    ]}
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt")
print(inputs.keys())
