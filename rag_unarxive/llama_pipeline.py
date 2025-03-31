print("Loading transformers...")
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
print("Loading torch...")
import torch
import os
import shutil


def get_llama_pipeline():
    print("\nLoading model...")
    # Set up device
    device = torch.device("cuda" if  torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load both model and tokenizer from the fine-tuned output directory
    #model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    
    model_in_cache_name = f"models--{model_name.split('/')[0]}--{model_name.split('/')[1]}"

    model_snapshot = "0e9e39f249a16976918f6564b8830bc894c89659" if model_name == "meta-llama/Llama-3.1-8B-Instruct" else "0cb88a4f764b7a12671c53f0838cd831a0843b95"

    hf_cache_path = os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub/")
    cache_path = os.path.join(os.path.expanduser(hf_cache_path), model_in_cache_name, f"snapshots/{model_snapshot}")
    tmp_base_path = "/tmp/s9650707/models/"
    tmp_path = os.path.join(tmp_base_path, model_in_cache_name)
    if os.path.exists(cache_path):
        print("Model exists in cache -> copy to tmp if necessary and use")
        if not os.path.exists(tmp_base_path):
            os.makedirs(tmp_base_path)
        if not os.path.exists(tmp_path):
            print("Copying model to tmp")
            shutil.copytree(cache_path, tmp_path)
        model_path = tmp_path
    else:
        print("Model does not exist in cache -> download")
        model_path = model_name
    
    # 8k context Finetuned model
    #model_path = "/data/horse/ws/s9650707-llm_workspace/scholaryllm_prot/os_train_data_finetune/model_checkpoints_100percent_lora32_shuffled_split/checkpoint-7970/"
    
    # 16k context Finetuned model
    model_path = "/data/horse/ws/s9650707-llm_workspace/scholaryllm_prot/os_train_data_finetune/model_checkpoints_100percent_lora32_shuffled_split_16k_context/checkpoint-3985/"
    print(f"Using model {model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16) #"Qwen/Qwen2-0.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_path) #"Qwen/Qwen2-0.5B-Instruct")  # Changed to load from fine-tuned path
    
    # Load LoRA weights
    #model = PeftModel.from_pretrained(base_model, model_path)
    model = base_model
    model = model.half().cuda()
    model.to(device)
    
    generator = pipeline(model=model, tokenizer=tokenizer, task="text-generation")
    return generator
