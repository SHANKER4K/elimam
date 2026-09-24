from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

model_id = "ALJIACHI/Mizan-Rerank-v1"
save_dir = "./models/reranker/onnx-mizan"

# Load and export to ONNX
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)

# Save ONNX model and tokenizer.json
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)
