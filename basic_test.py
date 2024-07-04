#Reusing the  model loading from setup.py 
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

cache_dir="./"


#Load the Peft Config from paper's CTD repo 
config = PeftConfig.from_pretrained("ppaudel/ctd-flant5-xxl")

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path,cache_dir="./")


#Change this if you want some other directory to use as cache 
base_model = AutoModelForSeq2SeqLM.from_pretrained("philschmid/flan-t5-xxl-sharded-fp16",load_in_8bit=True,device_map={"":0},cache_dir=cache_dir)
#Combine model 
model = PeftModel.from_pretrained(base_model, "ppaudel/ctd-flant5-xxl",device_map={"":0})
print("Model loaded . Now loading the datasets and running inference...")
print("---------")
def inference_random_samples(df_random):
    for index,row in df_random.iterrows():
        text=row["text"].replace("'","")
        input_text='''
                            Classify if a statement supports or refutes the scientific fact: {0}.
                                Statement: {1}.
                                Response: Refutes.
                                Statement: {2}.
                                Response: Supports.
                                Statement: {3}.
                                Response: 
                            '''
        fill_prompt=input_text.format(row['consensus'],row['refuting'],row['supporting'],row['text'])
        input_ids = tokenizer(fill_prompt, return_tensors="pt").input_ids.to("cuda")
        outputs = model.generate(input_ids=input_ids)
        final=tokenizer.decode(outputs[0])
        print("Model Output ",tokenizer.decode(outputs[0])," Ground truth ",row['stance'])

import pandas as pd
df = pd.read_csv("hf://datasets/ppaudel/dataset_covid_cq/dataset_covid_cq_triplet.csv")
random_samples=df.sample(10)
print("Dataset load test 1/4. Loading and inference against 10 random samples of COVID-CQ dataset")
print("----")
inference_random_samples(random_samples)
print("----")
df=pd.read_csv("hf://datasets/ppaudel/dataset_climate/dataset_climate_triplet.csv")
random_samples=df.sample(10)
print("Dataset load test 2/4. Loading and inference against 10 random samples of Climate Skepticism dataset")
inference_random_samples(random_samples)
###
print("----")
df=pd.read_csv("hf://datasets/ppaudel/electiondenial/dataset_electiondenial_deadvoters.csv")
random_samples=df.sample(10)
print("Dataset load test 3/4. Loading and inference against 10 random samples of Election denials dataset")
inference_random_samples(random_samples)
print("-----")
