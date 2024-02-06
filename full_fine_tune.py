from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer


model_name = "NousResearch/Llama-2-7b-chat-hf"
dataset_name = "mlabonne/guanaco-llama2-1k"
dataset = load_dataset(dataset_name, split="train")
device_map = {"": 0}

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # device_map=device_map
)
print("Model loaded!!!")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

import os

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "9994"  # modify if RuntimeError: Address already in use
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    optim="adamw_hf",
    logging_steps=1,
    log_level="debug",
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    report_to="tensorboard",
    skip_memory_metrics=False,
    gradient_checkpointing=True,
    deepspeed="ds_config_zero3.json",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

print("Training started!!")
trainer.train()
