import argparse

def parse_args():

    parser = argparse.ArgumentParser(description='Bayes-LM')

    parser.add_argument('--seed', 
                       default=1000, 
                       type=int)
    
    parser.add_argument('--add_plot', 
                       default=True, 
                       type=lambda x: (str(x).lower() == 'true'))

    ### Data loading / data preparation arguments -------------+

    parser.add_argument('--model_name', 
                       default='gaussian-gamma', 
                       help='Generating distribution: bernoulli, gaussian, gaussian-gamma, exponential, hmm, hypothesis.',
                       type=str)
    
    parser.add_argument('--target_name', 
                       default='posterior', 
                       help='',
                       type=str)
    
    parser.add_argument('--hypothesis_mode', 
                       default='easy', 
                       help='type [easy] or [hard] (equal width or unequal width).',
                       type=str)
    
    parser.add_argument('--random_init', 
                       default=False, 
                       type=lambda x: (str(x).lower() == 'true'))
    
    parser.add_argument('--generate_data', 
                       default=True, 
                       type=lambda x: (str(x).lower() == 'true'))
    
    parser.add_argument('--mlm', 
                       default=False, 
                       type=lambda x: (str(x).lower() == 'true'))
    
    parser.add_argument('--device', 
                       default='cuda', 
                       help='cpu / cuda.',
                       type=str)
        
    parser.add_argument('--N', 
                       default=10000, 
                       type=int)
    
    parser.add_argument('--V', 
                       default=1000, 
                       type=int)
    
    parser.add_argument('--T', 
                       default=500, 
                       type=int)
    
    parser.add_argument('--train_val_idx', 
                       default=4000, 
                       type=int)
    
    ### Probe training arguments -----------------------------+
    parser.add_argument('--use_linear_classifier', 
                       default=True, 
                       help='true for linear, anything else for non-linear (deep) classifier.',
                       type=lambda x: (str(x).lower() == 'true'))

    parser.add_argument('--token_num', 
                       default=-1, 
                       type=int)
    
    parser.add_argument('--token_unseen', 
                       default=0, 
                       type=int)
    
    parser.add_argument('--learning_rate', 
                       default=0.001, 
                       type=float)
    
    parser.add_argument('--batch_size', 
                       default=128, 
                       type=int)
    
    parser.add_argument('--accum_iter', 
                       default=1, 
                       type=int)
    
    parser.add_argument('--dropout',
                       default=0.1, 
                       type=float)
    
    parser.add_argument('--weight_decay',
                       default=0., 
                       type=float)
    
    parser.add_argument('--delta',
                       default=0.5, 
                       type=float)
    
    parser.add_argument('--probe_hidden_dim',
                       default=784, 
                       type=int)
    
    ### Misc arguments ---------------------------------+
    parser.add_argument('--general_description', 
                       default='LM-Bayes', 
                       type=str)
    
    ### Transformer arguments ------------------------+
    
    parser.add_argument('--d_model', 
                       default=128, 
                       type=int)
    
    parser.add_argument('--t_num_layers', 
                       default=3, 
                       type=int)
    
    parser.add_argument('--att_heads', 
                       default=8, 
                       type=int)
    
    parser.add_argument('--use_pos_enc', 
                       default=True, 
                       type=lambda x: (str(x).lower() == 'true'))

    args = parser.parse_args()
    return args