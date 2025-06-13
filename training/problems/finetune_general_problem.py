#finetune_general_problem.py
import re
import os
import argparse
import glob
import numpy as np

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

@registry.register_problem
class FinetuneGeneralProblem(text_problems.Text2TextProblem):

    @property
    def approx_vocab_size(self):
        return 2 ** 15 # ~32k

    @property
    def is_generate_per_split(self):
        # custom train/test split
        return True
    
    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("--t2t_usr_dir", type=str)
        parser.add_argument("--data_dir", type=str)
        parser.add_argument("--tmp_dir", type=str)
        parser.add_argument("--problem", type=str)
        parser.add_argument("--lang", type=str)
        parser.add_argument("--mode", type=str, default="basic", help="Mode: basic or extended")
        parser.add_argument("--token_err_prob", default=0.15, type=float, help="Probability of token error.")
        parser.add_argument("--token_std_dev", default=0.2, type=float, help="Standard deviation of token error.")
        parser.add_argument("--token_err_distribution", default="0.7_0.1_0.1_0.1_0", type=str, help="Space-separated error probabilities in format \"replace insert delete swap recase\".")

        parser.add_argument("--char_err_prob", default=0.05, type=float, help="Probability of char error.")
        parser.add_argument("--char_std_dev", default=0.01, type=float, help="Standard deviation of character error.")
        parser.add_argument("--char_err_distribution", default="0.25_0.25_0.25_0.25_0", type=str, help="Space-separated char-level error probabilities in format \"replace insert delete swap change_diacr\".")
        
        parser.add_argument("--data_ratio", default=1, type=int, help="Ratio of original vs artifical data, i.e. value of 50 means that 50 times more artificial data is used.")
        parser.add_argument("--additional_artificial_sentences", default=0, type=int, help="Number of artificially generated sentences.")
        parser.add_argument("--additional_wiki_sentences", default=0, type=int, help="Number of wiki sentences.")
        parser.add_argument("--additional_data_filtered", default="False", type=str, help="Are additional data filtered or not.")
        
        parser.add_argument("--input_sentence_file", type=str)
        parser.add_argument("--target_sentence_file", type=str)
        
        args = parser.parse_args()

        del data_dir
        del tmp_dir

        if args.mode == "extended":
            artificial_glob_pattern = '/app/malteseGEC/data/chunks/{}/extended/0.15-0.7_0.1_0.1_0.1_0-0.02-0.2_0.2_0.2_0.2_0.2_0/*.txt'.format(args.lang)
        else:
            # basic
            artificial_glob_pattern = '/app/malteseGEC/data/chunks/{}/basic/0.15-0.7_0.1_0.1_0.1_0-0.02-0.2_0.2_0.2_0.2_0.2_0/*.txt'.format(args.lang)
        
        artificial_chunks = sorted(glob.glob(artificial_glob_pattern))
        
        print("Mode: {}".format(args.mode))
        print("Looking for chunks with pattern: {}".format(artificial_glob_pattern))
        print("Found {} chunk files".format(len(artificial_chunks)))
        
        if dataset_split == problem.DatasetSplit.TRAIN:
            np.random.seed(42)

            # Load authentic training data
            original_data = []
            if os.path.exists(args.input_sentence_file) and os.path.exists(args.target_sentence_file):
                print("Loading authentic data from:")
                print("  Input: {}".format(args.input_sentence_file))
                print("  Target: {}".format(args.target_sentence_file))
                
                with open(args.input_sentence_file) as f1, open(args.target_sentence_file) as f2:
                    for l1, l2 in zip(f1, f2):
                        l1, l2 = l1.strip('\n'), l2.strip('\n')
                        if not l1 or not l2:
                            continue
                        original_data.append((l1, l2))
                
                print("Loaded {} authentic sentence pairs".format(len(original_data)))
            else:
                print("WARNING: Authentic data files not found!")

            # Load synthetic data from chunks
            artificial_lines = []
            for artificial_chunk in artificial_chunks:
                print("Loading chunk: {}".format(os.path.basename(artificial_chunk)))
                with open(artificial_chunk) as reader:
                    chunk_lines = reader.read().splitlines()
                    artificial_lines.extend(chunk_lines)
                    print("  Added {} lines from this chunk".format(len(chunk_lines)))
            
            print("Total artificial lines available: {}".format(len(artificial_lines)))

            # Generate authentic data based on ratio
            num_artificial_sentences = min(args.additional_artificial_sentences, len(artificial_lines))
            if len(original_data) > 0:
                num_original_data_cycles_to_generate = max(1, int((num_artificial_sentences / len(original_data)) / args.data_ratio))
                print('Generating {} cycles of original data ({} sentences)'.format(
                    num_original_data_cycles_to_generate, 
                    num_original_data_cycles_to_generate * len(original_data)))
                
                for _ in range(num_original_data_cycles_to_generate):
                    for l1, l2 in original_data:
                        yield {"inputs": l1, "targets": l2}

            # Generate synthetic data
            if num_artificial_sentences > 0 and len(artificial_lines) > 0:
                print('Generating {} artificial sentences'.format(num_artificial_sentences))
                
                # Randomly sample from available artificial sentences
                if num_artificial_sentences < len(artificial_lines):
                    permutation = np.random.permutation(len(artificial_lines))[:num_artificial_sentences]
                    selected_artificial_lines = [artificial_lines[i] for i in permutation]
                else:
                    # Use all available lines
                    selected_artificial_lines = artificial_lines
                    print("Warning: Requested {} sentences but only {} available".format(
                        num_artificial_sentences, len(artificial_lines)))

                for line in selected_artificial_lines:
                    chunks = line.split('\t')
                    if len(chunks) < 2:
                        print("Line in artificial data does not contain original and corrected version. Skipping it.")
                        print(chunks)
                        print(line)
                        continue
                    yield {"inputs": chunks[1], "targets": chunks[0]}
                    
        elif dataset_split == problem.DatasetSplit.EVAL:
            # Use authentic dev data
            dev_file = '/app/malteseGEC/data/authentic/splits/dev/parallel.tsv'
            print("Loading dev data from: {}".format(dev_file))
            
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
                print("Dev file not found: {}".format(dev_file))
                yield {"inputs": "Dev data not available", "targets": "Dev data not available"}
        else:
            # For test or other splits
            yield {"inputs": "Test data placeholder", "targets": "Test data placeholder"}