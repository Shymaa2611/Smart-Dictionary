from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(checkpoint_dir):
    token="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"  
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir,use_auth_token=token)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir,use_auth_token=token)
    return model, tokenizer

def generate_inference(model, tokenizer, input_text, max_length=50):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        num_beams=5, 
        early_stopping=True
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text


def get_word_antonym(word:str):
  checkpoint_dir = "Checkpoint/checkpoint"
  model, tokenizer = load_model_and_tokenizer(checkpoint_dir)
  input_text = f"the antonym of the word {word}"
  generated_text = generate_inference(model, tokenizer, input_text)
  return generated_text

