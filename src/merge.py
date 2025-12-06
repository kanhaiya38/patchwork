import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from constants import BASE_MODEL

base_model = BASE_MODEL
lora_path = "good_copies/run_final_meetingbank/continual/task_0_MeetingBank"
output_path = "merged-openllama-3b"

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model, torch_dtype=torch.bfloat16, device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, lora_path)

# Merge LoRA weights into base model
print("Merging LoRA into base model...")
model = model.merge_and_unload()

# Save merged model
model.save_pretrained(output_path)

# Save tokenizer (use base tokenizer)
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.save_pretrained(output_path)

print("Done. Merged model saved at:", output_path)
