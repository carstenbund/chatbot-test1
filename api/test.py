import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Use mounted model directory
MODEL_PATH = os.getenv("MODEL_PATH", "mistralai/Mistral-7B-Instruct-v0.2")

print(f"Loading model from: {MODEL_PATH}")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto"
)

# CLI Loop
print("\n Interactive CLI Mode: Type your prompt and press ENTER (or type 'exit' to quit)\n")
while True:
    prompt = input(" Enter prompt: ")
    if prompt.lower() in ["exit", "quit"]:
        print(" Exiting...")
        break

    print("\n⏳ Generating response...\n")
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_length=200)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"\n AI Response: {response}\n")
    except Exception as e:
        print(f"\n Error: {str(e)}\n")

