# backend/test_llm.py

import torch
# Monkey-patch for safetensors compatibility
if not hasattr(torch, "uint64"):
    torch.uint64 = torch.int64

from core.llm_hf import stream_hf_chat

if __name__ == "__main__":
    prompt = "User: Hello, how are you?\nAssistant:"
    print("Streaming response:")
    for token in stream_hf_chat(prompt):
        print(token, end="", flush=True)
