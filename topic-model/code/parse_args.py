import argparse

def parse_args():

    parser = argparse.ArgumentParser(description='Topic-LLM')

    ### Data loading / data preparation arguments -------------+
    parser.add_argument('--load_embeddings', 
                       default=True, 
                       type=lambda x: (str(x).lower() == 'true'))
    
    parser.add_argument('--load_embeddings_all', 
                       default=False, 
                       type=lambda x: (str(x).lower() == 'true'))
    
    parser.add_argument('--remove_header', 
                       default=True, 
                       type=lambda x: (str(x).lower() == 'true'))
    
    parser.add_argument('--use_eos_token', 
                       default=False, 
                       type=lambda x: (str(x).lower() == 'true'))
    
    parser.add_argument('--raw', 
                       default=False, 
                       type=lambda x: (str(x).lower() == 'true'))
    
    parser.add_argument('--pos', 
                       default=False, 
                       type=lambda x: (str(x).lower() == 'true'))
    
    parser.add_argument('--model_type', 
                       default='gpt2', 
                       help='gpt2 or llama2.',
                       type=str)
    
    parser.add_argument('--dataset', 
                       default='20ng', 
                       help='20ng / wiki / synthetic.',
                       type=str)
    
    ### Training arguments -----------------------------+
    parser.add_argument('--job', 
                       default='train_classifier', 
                       help='train_classifier or finetuning.',
                       type=str)

    parser.add_argument('--use_linear_classifier', 
                       default=True, 
                       help='true for linear, anything else for non-linear (deep) classifier.',
                       type=lambda x: (str(x).lower() == 'true'))

    parser.add_argument('--train_mode', 
                       default='classification', 
                       help='classification or bayesian.',
                       type=str)

    parser.add_argument('--token', 
                       default=-1, 
                       type=float)
    
    parser.add_argument('--learning_rate', 
                       default=0.0001, 
                       type=float)
    
    parser.add_argument('--batch_size', 
                       default=128, 
                       type=int)
    
    parser.add_argument('--accum_iter', 
                       default=1, 
                       type=int)
    
    parser.add_argument('--load_finetune', 
                       default='none', 
                       help='.',
                       type=str)
    
    parser.add_argument('--dropout',
                       default=0.1, 
                       type=float)
    
    parser.add_argument('--weight_decay',
                       default=0., 
                       type=float)
    
    parser.add_argument('--cls_hidden_dim',
                       default=768, 
                       type=int)
    
    ### Misc arguments ---------------------------------+
    parser.add_argument('--use_wandb', 
                       default=False, 
                       type=lambda x: (str(x).lower() == 'true'))

    parser.add_argument('--general_description', 
                       default='Topic-LLM on 20NG', 
                       type=str)
    
    ### Synthetic data arguments ------------------------+

    parser.add_argument('--llm_data_file_appendix', 
                       type=str)
    
    parser.add_argument('--cls_data_file_appendix', 
                       type=str)

    parser.add_argument('--seed', 
                       default=1000, 
                       type=int)
    
    parser.add_argument('--d_model', 
                       default=128, 
                       type=int)
    
    parser.add_argument('--N', 
                       default=10000, 
                       type=int)
    
    parser.add_argument('--V', 
                       default=1000, 
                       type=int)
    
    parser.add_argument('--M', 
                       default=100, 
                       type=int)
    
    parser.add_argument('--K', 
                       default=5, 
                       type=int)
    
    parser.add_argument('--num_layers', 
                       default=4, 
                       type=int)
    
    parser.add_argument('--use_pos_enc', 
                       default=True, 
                       type=lambda x: (str(x).lower() == 'true'))

    args = parser.parse_args()
    return args