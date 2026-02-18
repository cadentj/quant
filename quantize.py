import os
import base64
from io import BytesIO
import torch
from datasets import load_dataset
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot


# Load model.
model_id = "google/gemma-3-270m-it"
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="cuda:0",
    dtype="auto",
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

MAX_SEQUENCE_LENGTH = 2048
NUM_CALIBRATION_SAMPLES = 512

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))


def preprocess(example):
    return {"text": processor.apply_chat_template(example["messages"], tokenize=False)}


def tokenize(sample):
    return processor(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(preprocess)
ds = ds.map(tokenize, remove_columns=ds.column_names)

# # Oneshot arguments
# DATASET_ID = "neuralmagic/calibration"
# NUM_CALIBRATION_SAMPLES = 512
# MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
# ds = load_dataset(DATASET_ID, "LLM", split="train[:512]")
# ds = ds.shuffle(seed=42)

dampening_frac=0.05

# def data_collator(batch):
#     assert len(batch) == 1, "Only batch size of 1 is supported for calibration"
#     item = batch[0]
#     collated = {}
#     import torch


#     for key, value in item.items():
#         if isinstance(value, torch.Tensor):
#             collated[key] = value.unsqueeze(0)
#         elif isinstance(value, list) and isinstance(value[0][0], int):
#             # Handle tokenized inputs like input_ids, attention_mask
#             collated[key] = torch.tensor(value)
#         elif isinstance(value, list) and isinstance(value[0][0], float):
#             # Handle possible float sequences
#             collated[key] = torch.tensor(value)
#         elif isinstance(value, list) and isinstance(value[0][0], torch.Tensor):
#             # Handle batched image data (e.g., pixel_values as [C, H, W])
#             collated[key] = torch.stack(value)  # -> [1, C, H, W]
#         elif isinstance(value, torch.Tensor):
#             collated[key] = value
#         else:
#             print(f"[WARN] Unrecognized type in collator for key={key}, type={type(value)}")
    
#     return collated
   


# Recipe
recipe = [
    GPTQModifier(
        targets="Linear",
        ignore=["re:.*lm_head.*", "re:.*embed_tokens.*", "re:vision_tower.*", "re:multi_modal_projector.*"],
        sequential_targets=["Gemma3DecoderLayer"],
        dampening_frac=dampening_frac,
    )
]

SAVE_DIR="/workspace/quantized"
save_path = os.path.join(SAVE_DIR, model_id.split("/")[1] + "-W8A8")

# Perform oneshot
oneshot(
    model=model,
    tokenizer=model_id,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
    # data_collator=data_collator,
    output_dir=SAVE_DIR
)