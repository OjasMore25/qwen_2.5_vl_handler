import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from PIL import Image
from io import BytesIO
import base64

# skip strict version checks if sglang or other libs complain
os.environ["DISABLE_VERSION_CHECK"] = "1"


MODEL_NAME = "Qwen/Qwen2.5-VL-32B-Instruct"

print("[INFO] Loading Qwen2.5-VL-32B-Instruct model on A100...")


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,  # A100 supports bf16 natively
    device_map="auto",
    trust_remote_code=True
)
model.eval()
print("[INFO] Model loaded successfully.")


def run(job):
    try:
        prompt = job["input"].get("prompt", "")
        image_b64 = job["input"].get("image")

        if not image_b64:
            return {"error": "Missing 'image' in input"}

        # Decode the base64 image
        image = Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGB")

        # Prepare prompt+image in Qwen's expected format
        messages = [
            {"role": "user", "content": [
                {"image": image},
                {"text": prompt}
            ]}
        ]

        input_data = processor.tokenizer.from_list_format(messages)
        inputs = processor(input_data, return_tensors="pt").to(model.device)

        # Generate output
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )

        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"output": response_text}

    except Exception as e:
        return {"error": str(e)}
