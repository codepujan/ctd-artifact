from transformers import AutoModelForSeq2SeqLM


from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


cache_dir="./"
#Decrease batch size if you run into OOM errors 
BATCH_SIZE=50
tokenizer = AutoTokenizer.from_pretrained("philschmid/flan-t5-xxl-sharded-fp16",cache_dir=cache_dir)



from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


import pandas as pd



# In[4]:


from collections import defaultdict
from sklearn.metrics import f1_score,precision_score,recall_score,confusion_matrix

import numpy as np


# In[5]:


def compute_f1(precision,recall):
    return 2*(precision*recall)/(precision+recall)
def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# In[6]:


def compute_stats(results,results_gt):
    gts=[]
    preds=[]
    for index in range(0,len(results)):
        gts.append(0 if results_gt[index]["gt"].lower()=="refute" else 1)
        if "no" in results[index].lower() or "refute" in results[index].lower() or "ref" in results[index].lower():
            preds.append(0)
        elif "yes" in results[index].lower() or "support" in results[index].lower() or "sup" in results[index].lower():
            preds.append(1)
        else:
            print("Output space doesn't fall in support or refute ?",ground_truth.lower())
    print("F1 score",f1_score(gts,preds,average='weighted'))


# In[7]:


###
def infer_and_stat_full(df,input_model,batch_size):
    results_gt=[]
    prompts=[]
    idx=-1
    for index,row in df.iterrows():
      idx+=1
      if 1:
            text=row["text"].replace("'","")
            if 1:
                input_text='''
                    Classify if a statement supports or refutes the scientific fact: {0}.
                        Statement: {2}.
                        Response: Refutes.
                        Statement: {1}.
                        Response: Supports.
                        Statement: {3}.
                        Response:
                    '''
                prompts.append(input_text.format(row['consensus'],row['supporting'],row['refuting'],row['text']))
                results_gt.append({"text":text,"gt":row['stance']})
    #Now infer
    prompts_chunks= list(chunk(prompts, batch_size))
    results=[]
    for prompts in tqdm(prompts_chunks):
        input_ids = tokenizer(prompts, return_tensors="pt",padding=True,truncation=True,max_length=512).input_ids.to("cuda")
        outputs = input_model.generate(input_ids=input_ids)
        results_batch=tokenizer.batch_decode(outputs,skip_special_tokens=True)
        results.extend(results_batch)
        del outputs
        del results_batch
        torch.cuda.empty_cache()
        gc.collect()
    compute_stats(results,results_gt)


# In[12]:


from tqdm import tqdm
import gc,torch
model_id = "philschmid/flan-t5-xxl-sharded-fp16"

df=pd.read_csv("hf://datasets/ppaudel/dataset_climate/dataset_climate_triplet.csv")


print('-----')
print("Running bootstrapped CTD on Climate Skepticism dataset")

# huggingface hub model id

# load model from the hub
bootstrapped_model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto",cache_dir=cache_dir)

infer_and_stat_full(df,bootstrapped_model,BATCH_SIZE)


del bootstrapped_model
torch.cuda.empty_cache()
print("-----")
config = PeftConfig.from_pretrained("ppaudel/ctd-flant5-xxl")
base_model = AutoModelForSeq2SeqLM.from_pretrained("philschmid/flan-t5-xxl-sharded-fp16",load_in_8bit=True,device_map={"":0},cache_dir=cache_dir)
finetuned_model = PeftModel.from_pretrained(base_model, "ppaudel/ctd-flant5-xxl",device_map={"":0})
finetuned_model.eval()
print("Peft model loaded")

print("Now Running finetuned-CTD on Climate Skepticism dataset")

infer_and_stat_full(df,finetuned_model,BATCH_SIZE)


del finetuned_model
torch.cuda.empty_cache()



print("Running bootstrapped CTD on COVID-CQ")
# huggingface hub model id
model_id = "philschmid/flan-t5-xxl-sharded-fp16"

# load model from the hub
bootstrapped_model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto",cache_dir=cache_dir)
df = pd.read_csv("hf://datasets/ppaudel/dataset_covid_cq/dataset_covid_cq_triplet.csv")

infer_and_stat_full(df,bootstrapped_model,BATCH_SIZE)

del bootstrapped_model
torch.cuda.empty_cache()

print("Now Running finetuned-CTD on COVID-CQ")
print("-----")


config = PeftConfig.from_pretrained("ppaudel/ctd-flant5-xxl")
base_model = AutoModelForSeq2SeqLM.from_pretrained("philschmid/flan-t5-xxl-sharded-fp16",load_in_8bit=True,device_map={"":0},cache_dir=cache_dir)
finetuned_model = PeftModel.from_pretrained(base_model, "ppaudel/ctd-flant5-xxl",device_map={"":0})
finetuned_model.eval()

infer_and_stat_full(df,finetuned_model,BATCH_SIZE)

del finetuned_model
torch.cuda.empty_cache()


print("Running bootstrapped CTD on Stanceosaurus")
df=pd.read_csv("hf://datasets/ppaudel/twitter_factchecking_test/dataset_twitter_factchecking_triplet.csv")#

#
# load model from the hub
bootstrapped_model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto",cache_dir=cache_dir)

infer_and_stat_full(df,bootstrapped_model,BATCH_SIZE)

del bootstrapped_model
torch.cuda.empty_cache()

print("------")
print("Running finetuned CTD on Stanceosaurus")

config = PeftConfig.from_pretrained("ppaudel/ctd-flant5-xxl")
base_model = AutoModelForSeq2SeqLM.from_pretrained("philschmid/flan-t5-xxl-sharded-fp16",load_in_8bit=True,device_map={"":0},cache_dir=cache_dir)
finetuned_model = PeftModel.from_pretrained(base_model, "ppaudel/ctd-flant5-xxl",device_map={"":0})
finetuned_model.eval()

infer_and_stat_full(df,finetuned_model,BATCH_SIZE)
