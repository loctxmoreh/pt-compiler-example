import torch 
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

device = "cuda"
model_name = "meta-llama/Llama-3.1-8B"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
model = torch.compile(model, mode="max-autotune")

prompt = "Once upon a time, "
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.forward(**inputs)

logits = outputs.logits
next_token_logits = logits[:, -1, :]
next_token_id = torch.argmax(next_token_logits, dim=-1)

generated_ids = torch.cat([inputs["input_ids"], next_token_id.unsqueeze(-1)], dim=-1)

generated_text = tokenizer.decode(generated_ids.squeeze(), skip_special_tokens=True)
print(generated_text)
