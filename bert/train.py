import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, AdamW
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
model.to(device)

# Load train data
train_path = "train.json"

#train_path = "train-v1.1.json"
with open(train_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Flatten SQuAD-style JSON into lists of contexts, questions, and answers
contexts = []
questions = [] 
answers = [] # ie: answers[i] = {"answer_start": 515, "text": "Saint Bernadette Soubirous"}
for article in data["data"]:
    for paragraph in article["paragraphs"]:
        context = paragraph["context"]
        for qa in paragraph["qas"]:
            question = qa["question"]
            if len(qa["answers"]) > 0:
                answer = qa["answers"][0]
                contexts.append(context)
                questions.append(question)
                answers.append(answer)

# Tokenize and create Dataset
class QADataset(Dataset):
    def __init__(self, contexts, questions, answers, tokenizer):
        self.encodings = self.preprocess(contexts, questions, answers, tokenizer)

    def preprocess(self, contexts, questions, answers, tokenizer):
        # dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping'])
        encodings = tokenizer(
            questions,
            contexts,
            truncation=True,
            padding="max_length",
            max_length=384,
            return_offsets_mapping=True
        )

        start_positions = [] # token index of answer start
        end_positions = []   # token index of answer end

        for i, offsets in enumerate(encodings["offset_mapping"]):
            answer = answers[i]
            start_char = answer["answer_start"]
            end_char = start_char + len(answer["text"])
            sequence_ids = encodings.sequence_ids(i)

            # Get context token range
            context_tokens = [idx for idx, s in enumerate(sequence_ids) if s == 1]
            if not context_tokens:
                start_positions.append(0)
                end_positions.append(0)
                continue

            context_start, context_end = context_tokens[0], context_tokens[-1]

            # If answer not fully inside the context span, mark as no-answer (CLS)
            if not (start_char < offsets[context_end][1] and start_char > offsets[context_start][0] and end_char < offsets[context_end][1] and end_char > offsets[context_start][0]):
                start_positions.append(0)
                end_positions.append(0)
                continue

            # Find start token index
            start_pos = context_start
            for idx in range(context_start, context_end + 1):
                if offsets[idx][0] <= start_char < offsets[idx][1]:
                    start_pos = idx
                    break

            # Find end token index
            end_pos = context_end
            for idx in range(start_pos, context_end + 1):
                if offsets[idx][0] < end_char <= offsets[idx][1]:
                    end_pos = idx
                    break

            start_positions.append(start_pos)
            end_positions.append(end_pos)

        encodings.update({
            "start_positions": start_positions,
            "end_positions": end_positions
        })
        encodings.pop("offset_mapping")
        return encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])

# Create DataLoader
train_dataset = QADataset(contexts, questions, answers, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Train loop
optimizer = AdamW(model.parameters(), lr=3e-5)

model.train()
# Surprisingly, increasing epoch to 3 does not improve performance much
for epoch in range(1):
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, _, _ = model(**batch)
        loss.backward()
        optimizer.step()
        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())

# Save model
torch.save(model.state_dict(), "bert_qa.pt")
print("Model saved to bert_qa.pt")
