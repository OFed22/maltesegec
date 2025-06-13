import re
import os
import argparse
import glob
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer
from tensor2tensor.utils import metrics

@registry.register_problem
class ArtificialErrors(text_problems.Text2TextProblem):
    @property
    def approx_vocab_size(self):
        return 2 ** 15 # ~32k
    @property
    def is_generate_per_split(self):
        # custom train/test split
        return True
    def eval_metrics(self):
        """Metrics to evaluate during eval."""
        return [
            metrics.Metrics.ACC,
            metrics.Metrics.ACC_TOP5,
            metrics.Metrics.ACC_PER_SEQ,
            metrics.Metrics.APPROX_BLEU,
            metrics.Metrics.NEG_LOG_PERPLEXITY,
        ]
    
    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("--t2t_usr_dir", type=str)
        parser.add_argument("--data_dir", type=str)
        parser.add_argument("--tmp_dir", type=str)
        parser.add_argument("--problem", type=str)
        parser.add_argument("--lang", type=str)
        parser.add_argument("--mode", type=str, default="basic", help="Mode: basic, extended, or asr")
        parser.add_argument("--token_err_prob", default=0.15, type=float, help="Probability of token error.")
        parser.add_argument("--token_std_dev", default=0.2, type=float, help="Standard deviation of token error.")
        parser.add_argument("--token_err_distribution", default="0.7_0.1_0.1_0.1", type=str, help="Space-separated error probabilities in format \"replace insert delete swap\".")
        parser.add_argument("--char_err_prob", default=0.05, type=float, help="Probability of char error.")
        parser.add_argument("--char_std_dev", default=0.01, type=float, help="Standard deviation of character error.")
        parser.add_argument("--char_err_distribution", default="0.25_0.25_0.25_0.25_0", type=str, help="Space-separated char-level error probabilities in format \"replace insert delete swap change_diacr\".")
        # Extended mode parameters
        parser.add_argument("--extended_token_distribution", default="0.65_0.05_0.05_0.05_0.1_0.1", type=str, help="Extended token distribution")
        parser.add_argument("--reverse_prob", default=0.02, type=float, help="Probability of reverse operation")
        parser.add_argument("--mono_prob", default=0.02, type=float, help="Probability of mono operation")
        args = parser.parse_args()
        del data_dir
        del tmp_dir
        
        if dataset_split == 'train':
            # Handle special pre-existing datasets
            if args.mode in ['debattista', 'busuttil']:
                data_file = f'/app/malteseGEC/data/chunks/{args.lang}/{args.mode}/preexisting_data.txt'
                print(f'Training with pre-existing {args.mode} dataset: {data_file}')
                
                if os.path.exists(data_file):
                    with open(data_file, encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip('\n')
                            if not line:
                                continue
                            chunks = line.split('\t')
                            if len(chunks) < 2:
                                print(f"Bad format at line {line_num}. Skipping.")
                                continue
                            # For pre-existing datasets: inputs=source, targets=corrected
                            yield {"inputs": chunks[0], "targets": chunks[1]}
                else:
                    print(f"ERROR: Pre-existing dataset not found: {data_file}")
                    yield {"inputs": f"No {args.mode} data", "targets": f"No {args.mode} data"}
            else:
                # Regular synthetic data handling
                glob_pattern = '/app/malteseGEC/data/chunks/{}/{}/0.15-0.7_0.1_0.1_0.1_0-0.02-0.2_0.2_0.2_0.2_0.2_0/*.txt'.format(args.lang, args.mode)
                
                print('Training - mode: {} - glob_pattern: {}'.format(args.mode, glob_pattern))
                train_files = glob.glob(glob_pattern)
                
                print(f"Found {len(train_files)} training files:")
                for f in train_files:
                    print(f"  - {f}")
                
                for train_file in train_files:
                    print(f"Processing {train_file}")
                    with open(train_file, encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip('\n')
                            if not line:
                                continue
                            chunks = line.split('\t')
                            if len(chunks) < 2:
                                print("Provided file {} seems to have data in bad format at line {}. Skipping the line.".format(train_file, line_num))
                                print(chunks)
                                print(line)
                                continue
                            yield {"inputs": chunks[1], "targets": chunks[0]}
                        
        elif dataset_split == 'dev':
            # Use authentic dev data for evaluation
            dev_file = '/app/malteseGEC/data/authentic/splits/dev/parallel.tsv'
            print(f"Loading dev data from: {dev_file}")
            
            if os.path.exists(dev_file):
                with open(dev_file, encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip('\n')
                        if not line:
                            continue
                        chunks = line.split('\t')
                        if len(chunks) < 2:
                            print("Dev file {} has bad format at line {}. Skipping.".format(dev_file, line_num))
                            continue
                        yield {"inputs": chunks[0], "targets": chunks[1]}
            else:
                print(f"Dev file not found: {dev_file}")
                yield {"inputs": "Dev data not available", "targets": "Dev data not available"}
                
        else:
            print(f"Using minimal data for split: {dataset_split}")
            yield {"inputs": "Test data placeholder", "targets": "Test data placeholder"}
            
@registry.register_hparams
def transformer_base_single_gpu_with_dropout():
    """Base parameters for Transformer model with custom dropouts."""
    hparams = transformer.transformer_base_single_gpu()
   
    # Add custom hyperparameters
    hparams.add_hparam("input_word_dropout", 0.2)
    hparams.add_hparam("target_word_dropout", 0.1)
    hparams.add_hparam("edit_weight", 3.0)
   
    return hparams