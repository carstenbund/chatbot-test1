from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Use mounted model directory
MODEL_PATH = "mistralai/Mistral-7B-Instruct-v0.3"

# Load Mistral model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto"
)

# Define request model
class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 200

@app.get("/")
def read_root():
    return {"message": "Mistral 7B API is running"}

@app.post("/generate")
def generate_text(request: PromptRequest):
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_length=request.max_tokens)
        return {"response": tokenizer.decode(output[0], skip_special_tokens=True)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
