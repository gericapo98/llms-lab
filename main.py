# ~/repos/llms-lab/main.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # safer on Apple GPU

MODEL_DIR = "models/deepseek-math-7b-instruct"
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {device}")

tok = AutoTokenizer.from_pretrained(MODEL_DIR)

# Build a chat-formatted prompt
messages = [
    {"role": "system", "content": "You are a helpful math assistant."},
    {"role": "user",   "content": "Solve for real x. x=x+5x^2 \\boxed{}."}
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Load model (start with fp16 on MPS; if it errors, try float32)
dtype = torch.float16 if device == "mps" else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=dtype,
    low_cpu_mem_usage=True
).to(device)

# Generate
inputs = tok(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=192,
        temperature=0.2,
        do_sample=False,
        pad_token_id=tok.eos_token_id
    )

text = tok.decode(out[0], skip_special_tokens=True)
print("\n==== OUTPUT ====\n")
print(text)

