## LLaMA 2 & Orca 2 Tutorial


**Step 1.** Register for LLaMA download at [https://ai.meta.com/resources/models-and-libraries/llama-downloads/](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).

**Step 2.** Install LLaMA models. 
```
$ git clone https://github.com/facebookresearch/llama
$ cd llama
$ ./download.sh   ## Use the email-received approval URL. 
```
CAUTION: Do not copy the email's link directly, but click the link and then copy the link on the address bar. Ensure to re-clone the repository and do this all at once without making any mistake. 

**Step 3.** Install the latest NVidia driver.
```
$ sudo ubuntu-drivers autoinstall
$ sudo apt install nvidia-driver-545
$ sudo reboot now
$ nvidia-smi   # Check if Nvidia is reachable
```

CAUTION: If you get an error for `sudo ubuntu-drivers autoinstall`:  
``` 
$ sudo vi /usr/lib/python3/dist-packages/UbuntuDrivers/detect.py
```
Fix  835 line's `version = int(package_name.split('-')[-1])`, change to `version = int(package_name.split('-')[-2])` (or to `[2]`)


**Step 4.** Install essential software.
```
$ sudo apt update
$ sudo apt upgrade
$ sudo apt install python3-pip  nvidia-cuda-toolkit
$ cd llama
$ pip3 install torch numpy scipy fire fairscale sentencepiece accelerate transformers
$ pip3 install -e .
$ python3
 > import torch
 > torch.cuda.is_available()
 > torch.cuda.device_count()
 > torch.cuda.current_device()
 > torch.cuda.device(0)
 > torch.cuda.get_device_name(0)
```

**Step 5.** Run the LLaMA example chatbot.
```
### 7B Chat Model (1 GPU needed)
$ python3 -m torch.distributed.run --nproc_per_node 1 example_chat_completion.py     --ckpt_dir llama-2-7b-chat/     --tokenizer_path tokenizer.model     --max_seq_len 512 --max_batch_size 6

### 13B Chat Model (2 GPUs needed)
$ python3 -m torch.distributed.run --nproc_per_node 2 example_chat_completion.py     --ckpt_dir llama-2-7b-chat/     --tokenizer_path tokenizer.model     --max_seq_len 512 --max_batch_size 6

### 70B Chat Model (8 GPUs needed)
$ python3 -m torch.distributed.run --nproc_per_node 8 example_chat_completion.py     --ckpt_dir llama-2-7b-chat/     --tokenizer_path tokenizer.model     --max_seq_len 512 --max_batch_size 6

### 7B Text Completion Model (1 GPU needed)
$ python3 -m torch.distributed.run --nproc_per_node 1 example_text_completion.py --ckpt_dir llama-2-7b/ --tokenizer_path tokenizer.model --max_seq_len 128 --max_batch_size 4

### 13B Text Completion Model (2 GPUs needed)
$ python3 -m torch.distributed.run --nproc_per_node 2 example_text_completion.py --ckpt_dir llama-2-13b/ --tokenizer_path tokenizer.model --max_seq_len 128 --max_batch_size 4

### 70B Text Completion Model (4 GPUs needed)
$ python3 -m torch.distributed.run --nproc_per_node 8 example_text_completion.py --ckpt_dir llama-2-70b/ --tokenizer_path tokenizer.model --max_seq_len 128 --max_batch_size 4
```


### Running LLaMA2 with Hugging Face (Google Colab or Local Setup)
- Model URL: [https://huggingface.co/meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
- Tutorial: [https://colab.research.google.com/drive/1SQmK0GYz34RGVlOnL5YMkdm7hXD6OjQT?usp=sharing](https://colab.research.google.com/drive/1SQmK0GYz34RGVlOnL5YMkdm7hXD6OjQT?usp=sharing)
- Minimum GPU RAM Size: 150GB ~ 180GB
- Run `Chatbot_LLaMa_2.ipynb` on Google Colab. Ensure to set up Runtime --> Change the runtime type --> some GPU.
- If you want local execution:
```
$ python3 llama2-hf-inference.py   # replace the Hugging Face token by yours at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
```



### Running Orca2 with Hugging Face (Google Colab or Local Setup)
- Model URL: [https://huggingface.co/microsoft/Orca-2-13b](https://huggingface.co/microsoft/Orca-2-13b)
- Tutorial: [https://m0nads.wordpress.com/2023/11/27/orca-2-on-colab/](https://m0nads.wordpress.com/2023/11/27/orca-2-on-colab/)
- Minimum GPU RAM Size: 15GB
- If you want local execution:
```
$ python3 orca2-hf-inference.py   # replace the Hugging Face token by yours at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)


### Resources
- [Hugging Face Library's Quantization](https://huggingface.co/docs/transformers/main_classes/quantization) for smaller model sizes
