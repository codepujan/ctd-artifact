from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


#Load the Peft Config from paper's CTD repo 
config = PeftConfig.from_pretrained("ppaudel/ctd-flant5-xxl")

#Change this if you want some other directory to use as cache 
cache_dir="./"
base_model = AutoModelForSeq2SeqLM.from_pretrained("philschmid/flan-t5-xxl-sharded-fp16",load_in_8bit=True,device_map={"":0},cache_dir=cache_dir)
#Combine model 
model = PeftModel.from_pretrained(base_model, "ppaudel/ctd-flant5-xxl",device_map={"":0})
#
model.eval()

print("Peft model loaded. You can proceed to running basic_test.py to make sure all datasets are loaded and model can run inference.")
