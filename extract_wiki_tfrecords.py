#!/usr/bin/env python3
"""
Extract Wikipedia revision data from TFRecords back to text format for finetuning.
"""

import tensorflow as tf
import os
import glob
from tensor2tensor.data_generators.text_encoder import SubwordTextEncoder

# Enable eager execution for TensorFlow 1.x
if hasattr(tf, 'enable_eager_execution'):
    tf.enable_eager_execution()

def extract_tfrecords_to_text(tfrecord_pattern, vocab_file, output_src, output_trg):
    """Extract parallel sentences from TFRecords."""
    
    # Load the subword encoder
    print(f"Loading vocabulary from {vocab_file}")
    
    # Check different possible vocab file names
    if not os.path.exists(vocab_file):
        # Try without .subwords extension
        base_vocab = vocab_file.replace('.subwords', '')
        if os.path.exists(base_vocab):
            print(f"Using vocabulary file: {base_vocab}")
            subword_encoder = SubwordTextEncoder(base_vocab)
        else:
            # Try alternative vocab paths
            alt_vocab = vocab_file.replace("wiki_revision", "artificial_errors")
            if os.path.exists(alt_vocab):
                print(f"Using alternative vocab: {alt_vocab}")
                subword_encoder = SubwordTextEncoder(alt_vocab)
            else:
                # List available vocab files
                vocab_dir = os.path.dirname(vocab_file)
                print(f"Available vocab files in {vocab_dir}:")
                if os.path.exists(vocab_dir):
                    for f in os.listdir(vocab_dir):
                        if 'vocab' in f:
                            print(f"  - {f}")
                raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")
    else:
        subword_encoder = SubwordTextEncoder(vocab_file)
    
    # Find all TFRecord files
    tfrecord_files = sorted(glob.glob(tfrecord_pattern))
    print(f"Found {len(tfrecord_files)} TFRecord files")
    
    count = 0
    with open(output_src, 'w', encoding='utf-8') as src_file, \
         open(output_trg, 'w', encoding='utf-8') as trg_file:
        
        for tfrecord_file in tfrecord_files:
            print(f"Processing {tfrecord_file}")
            
            # Create iterator for the TFRecord file
            dataset = tf.data.TFRecordDataset(tfrecord_file)
            
            for raw_record in dataset:
                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())
                
                # Extract features
                features = example.features.feature
                
                # Get inputs and targets
                if 'inputs' in features and 'targets' in features:
                    inputs = features['inputs'].int64_list.value
                    targets = features['targets'].int64_list.value
                    
                    # Decode from subwords to text
                    input_text = subword_encoder.decode(inputs)
                    target_text = subword_encoder.decode(targets)
                    
                    # Write to files
                    src_file.write(input_text + '\n')
                    trg_file.write(target_text + '\n')
                    
                    count += 1
                    if count % 1000 == 0:
                        print(f"Processed {count} examples")
    
    print(f"Extraction complete. Total examples: {count}")
    print(f"Source file: {output_src}")
    print(f"Target file: {output_trg}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfrecord_pattern", 
                        default="/home/fed/malteseGEC/training/t2t_data/wiki_revision/wiki_revision-train-*",
                        help="Pattern for TFRecord files")
    parser.add_argument("--vocab_file",
                        default="/home/fed/malteseGEC/training/t2t_data/wiki_revision/vocab.wiki_revision.strip.32768",
                        help="Vocabulary file path")
    parser.add_argument("--output_dir",
                        default="/app/malteseGEC/data/authentic/wikipedia",
                        help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Output file paths
    output_src = os.path.join(args.output_dir, "wiki_src.txt")
    output_trg = os.path.join(args.output_dir, "wiki_trg.txt")
    
    # Extract data
    extract_tfrecords_to_text(args.tfrecord_pattern, args.vocab_file, output_src, output_trg)

if __name__ == "__main__":
    main()