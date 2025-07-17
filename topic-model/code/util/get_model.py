import torch
import transformers

def get_model(config):

    if config['model_type'] == 'gpt2':

        tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
        model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token 

    if config['model_type'] == 'gpt2-random':

        tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
        configuration = transformers.GPT2Config()
        model = transformers.GPT2LMHeadModel(configuration)
        tokenizer.pad_token = tokenizer.eos_token 

    elif config['model_type'] == 'gpt2-large':
        
        tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2-large")
        model = transformers.GPT2LMHeadModel.from_pretrained("gpt2-large") 
        tokenizer.pad_token = tokenizer.eos_token

    elif config['model_type'] == 'gpt2-medium':
        
        tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2-medium")
        model = transformers.GPT2LMHeadModel.from_pretrained("gpt2-medium") 
        tokenizer.pad_token = tokenizer.eos_token

    elif config['model_type'] == 'bert':
        tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        model = transformers.BertLMHeadModel.from_pretrained("bert-base-uncased")
    
    elif config['model_type'] == 'bert-large':
        tokenizer = transformers.BertTokenizer.from_pretrained('bert-large-uncased')
        model = transformers.BertLMHeadModel.from_pretrained("bert-large-uncased")
    
    elif config['model_type'] == 'llama2-chat':

        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model_id = "NousResearch/Llama-2-7b-chat-hf"

        model_config = transformers.AutoConfig.from_pretrained(
            model_id
        )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto'
        )

        print('model loaded')

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    elif config['model_type'] == 'llama2':

        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model_id = "NousResearch/Llama-2-7b-hf"

        model_config = transformers.AutoConfig.from_pretrained(
            model_id
        )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto'
        )

        print('model loaded')

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    return model, tokenizer