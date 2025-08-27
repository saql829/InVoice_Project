# backend/core/llm_model.py
import os, threading
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

load_dotenv()
LLM_MODEL = os.getenv("LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
HF_API_KEY = os.getenv("HF_API_KEY")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, token=HF_API_KEY, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    token=HF_API_KEY,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
)
model.to(DEVICE)
model.eval()


def build_prompt(messages):
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = ""
    for m in messages:
        role = m["role"].capitalize()
        prompt += f"{role}: {m['content']}\n"
    prompt += "Assistant: "
    return prompt


def stream_llm(messages, max_new_tokens=128, temperature=0.7, top_p=0.9):
    """
    Stream response from LLM, yield string chunks (never generator objects).
    """
    prompt = build_prompt(messages)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        inputs=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        top_p=top_p,
        streamer=streamer,
        eos_token_id=tokenizer.eos_token_id,
    )

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    for new_text in streamer:
        # 🔥 ensure string
        yield str(new_text)


def generate_response(user_input: str) -> str:
    """Non-streaming single shot for /chat/voice."""
    msgs = [
        {"role": "system", "content": "You are a concise helpful assistant."},
        {"role": "user", "content": user_input},
    ]
    out = []
    for chunk in stream_llm(msgs, max_new_tokens=128, temperature=0.7, top_p=0.9):
        out.append(chunk)
    return "".join(out).strip()
