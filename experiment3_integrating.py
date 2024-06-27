#!/usr/bin/env python


print("Just Making sure GPU is available through Nvidia SMI ! ")


# In[1]:




# In[17]:


from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from tqdm import tqdm
import gc,torch


# In[18]:


config = PeftConfig.from_pretrained("ppaudel/ctd-flant5-xxl")
base_model = AutoModelForSeq2SeqLM.from_pretrained("philschmid/flan-t5-xxl-sharded-fp16",load_in_8bit=True,device_map={"":0},cache_dir="./")
model = PeftModel.from_pretrained(base_model, "ppaudel/ctd-flant5-xxl",device_map={"":0})
model.eval()

print("Peft model loaded")


# In[19]:


tokenizer = AutoTokenizer.from_pretrained("philschmid/flan-t5-xxl-sharded-fp16",cache_dir="./")


# In[2]:


### Lambretta stuff


# In[81]:


def compute_f1(precision,recall):
    return 2*(precision*recall)/(precision+recall)
def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# In[107]:


from collections import defaultdict
from sklearn.metrics import f1_score,precision_score,recall_score,confusion_matrix

import numpy as np

def false_negative_rate(gts, preds):
    gts = np.array(gts)
    preds = np.array(preds)
    
    total_refuting = np.sum(gts == 0)
    false_negatives = np.sum((gts == 0) & (preds == 1))
    
    return false_negatives / total_refuting if total_refuting > 0 else 0
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
    print("False Detection Rate",1-precision_score(gts,preds))
    print("False Negative Rate",false_negative_rate(gts,preds))





# In[111]:


def compute_lambretta_stats(df):
    print("Displaying stats of Election Denial claims ")#Todisplay claim 
    lambretta_fp=len(df[df['stance']=='Support'])/len(df)
    print("False positive of Lambretta without CTD",lambretta_fp)
    print("F1 score of Lambretta without CTD ",compute_f1(1-lambretta_fp,1))#Recall of Lambretta without CTD is always 1 


# In[113]:

def infer_and_stat(df):
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
                            Statement: {1}.
                            Response: Refutes.
                            Statement: {2}.
                            Response: Supports.
                            Statement: {3}.
                            Response: 
                        '''
                #
                prompts.append(input_text.format(row['consensus'],row['refuting'],row['supporting'],row['text']))
                results_gt.append({"text":text,"gt":row['stance']})
    #Now infer
    prompts_chunks= list(chunk(prompts, 50))
    results=[]
    for prompts in tqdm(prompts_chunks):
        input_ids = tokenizer(prompts, return_tensors="pt",padding=True,truncation=True,max_length=512).input_ids.to("cuda")
        outputs = model.generate(input_ids=input_ids)
        results_batch=tokenizer.batch_decode(outputs,skip_special_tokens=True)
        results.extend(results_batch)
        del outputs
        del results_batch
        torch.cuda.empty_cache()
        gc.collect()
    compute_stats(results,results_gt)


# In[114]:


# In[112]:
# In[116]:


print("Evaluating Claim #1 GA suitcase of ballots ")


# In[110]:


import pandas as pd
df=pd.read_csv("hf://datasets/ppaudel/electiondenial/dataset_electiondenial_georgia_suitcases.csv")

compute_lambretta_stats(df)

infer_and_stat(df)


# In[117]:


print("Evaluating Claim #2 Dead Voters in MI  ")


# In[118]:


import pandas as pd
df=pd.read_csv("hf://datasets/ppaudel/electiondenial/dataset_electiondenial_deadvoters.csv")
compute_lambretta_stats(df)
infer_and_stat(df)


# In[119]:


print("Evaluating Claim #3 WI Voter Turnout above 90%")
import pandas as pd
df=pd.read_csv("hf://datasets/ppaudel/electiondenial/dataset_electiondenial_wisconsinturnout.csv")
compute_lambretta_stats(df)
infer_and_stat(df)
