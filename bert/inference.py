import json
import torch
from transformers import BertTokenizerFast
from model import BertForQA
from tqdm import tqdm

# Configurations
model_name = "bert-base-cased"
layers = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForQA(model_name,layers)
model.load_state_dict(torch.load("bert_qa.pt", map_location=device))
model.to(device)
model.eval()

# Load data
dev_path = "test.json" 
#dev_path = "dev-v1.1.json" 
with open(dev_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Flatten SQuAD-style JSON into lists of contexts, questions, ids
contexts = []
questions = [] 
ids = []
for article in data["data"]:
    for paragraph in article["paragraphs"]:
        context = paragraph["context"]
        for qa in paragraph["qas"]:
            question = qa["question"]
            contexts.append(context)
            questions.append(question)
            ids.append(qa["id"])

# Model inference
predictions = {}

for i in tqdm(range(len(contexts))):
    context = contexts[i]
    question = questions[i]
    qid = ids[i]

    inputs = tokenizer.encode_plus(
        question,
        context,
        truncation=True,
        padding="max_length",
        max_length=384,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        _, start_logits, end_logits = outputs
        
        start_probs = torch.flatten(torch.nn.functional.softmax(start_logits,dim=1))
        end_probs = torch.flatten(torch.nn.functional.softmax(end_logits,dim=1))
        
        seq_len = start_probs.size(0)

        # Compute joint probabilities
        joint_probs = torch.outer(start_probs, end_probs)  # [seq_len, seq_len]

        # Mask invalid spans: only allow end >= start
        joint_probs = torch.triu(joint_probs)

        # Optionally limit span length
        max_answer_length = 30
        if max_answer_length is not None:
            for i in range(seq_len):
                joint_probs[i, i + max_answer_length :] = 0.0

        # Find the best span (i, j)
        flat_index = torch.argmax(joint_probs)
        start_idx, end_idx = divmod(flat_index.item(), seq_len)

        # If start > end, fix end = start
        if start_idx > end_idx:
            end_idx = start_idx

        #start_idx = torch.argmax(start_probs, dim=-1).item()
        #end_idx = torch.argmax(end_probs, dim=-1).item()

        #if end_idx < start_idx:
        #    end_idx = start_idx

        tokens = inputs["input_ids"][0][start_idx : end_idx + 1]
        answer = tokenizer.decode(tokens, skip_special_tokens=True)
        
        predictions[qid] = answer

# Save predictions
with open("bert_predictions.json", "w", encoding="utf-8") as f:
    json.dump(predictions, f, indent=2, ensure_ascii=False)

print("Predictions saved to bert_predictions.json")
