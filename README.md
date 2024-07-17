
# CTD-ARTIFACT
This repository contains companion scripts to run artifact evaluation for our Usenix 24 paper. 
 --
 #### Hardware dependencies
 The CTD model needs proper environment setup to run GPU based inference. 
 Therefore, it is recommended to run the evaluation experiments on environments with following configuration met
 

 1. CUDA Toolkit (v11.3)
 2. CUDNN (v8.2)
 3. GCC (v9.3.0)
 ---
 #### GPU memory requirements 
 The backbone model used by CTD is Google's FLAN-T5-XXL. We use a sharded version of this model loaded in 8-bit  configuration to make sure inference is possible in consumer hardware.
 It is recommended to run the command `nvidia-smi` on the development environment to make sure appropriate GPU memory requirements are met.

---

### Installing Python Packages
After configuring  software dependencies, please run `pip install -r requirements.txt` to install all dependencies are installed. 
You can run `python basic_test.py` to make sure the necessary dependencies are met and move forward with the individual experiments 

 1. `experiment1_bootstrapping.py`
 2. `experiment2_evaluating.py`
 3. `experiment3_integrating.py`

 ---
 Issues you might run into and workarounds 

Issue #1 CUDA OOM error
    
 Workaround: Try decreasing the value of `BATCH_SIZE` on the Python scripts. For 24G of memory, batch size of 50 should work. 

 ---
### Citing The Paper
```
@inproceedings {sec24:ctd,
    title = {Enabling Contextual Soft Moderation on Social Media through Contrastive Textual Deviation},
    booktitle = {33rd USENIX Security Symposium (USENIX Security)},
    year = {2024},
    author={Pujan Paudel, Mohammad Hammas Saeed, Rebecca Auger, Chris Wells, and Gianluca Stringhini}
}

