# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from datetime import datetime

datetime = str(datetime.now().strftime("%Y-%m-%d/%H-%M-%S"))


logging.basicConfig(filename=f'outputs/{datetime}_testing.log' ,
                    format='%(asctime)s | %(levelname)s: %(message)s', level=logging.INFO)


tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b", cache_dir='cache/')
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b", cache_dir='cache/')


text = "The manufacturer of Colt King Cobra was who?"


input_ids = tokenizer.encode(text, return_tensors="pt") 


output = model.generate(input_ids.to(model.device), num_return_sequences=1,max_new_tokens=22 )  
print(output)
output_text = tokenizer.decode(output[0][int(input_ids.shape[1]):], skip_special_tokens=True)
logging.info(f" \n ==>> output: \n {output_text}")




model1 = AutoModelForCausalLM.from_pretrained("checkpoints/test_model")
logging.info('\n loaded edited model ')
output = model.generate(input_ids.to(model.device), num_return_sequences=1,max_new_tokens=22 )  
output_text = tokenizer.decode(output[0][int(input_ids.shape[1]):], skip_special_tokens=True)
logging.info(f"\n \n ==>> output from edited model: \n {output_text}")


