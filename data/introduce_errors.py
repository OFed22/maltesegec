import argparse
import fileinput
import string
import os
import sys

import aspell
import numpy as np


allowed_source_delete_tokens = [',', '.', '!', '?']

czech_diacritics_tuples = [('a', 'á'), ('c', 'č'), ('d', 'ď'), ('e', 'é', 'ě'), ('i', 'í'), ('n', 'ň'), ('o', 'ó'), ('r', 'ř'), ('s', 'š'),
                           ('t', 'ť'), ('u', 'ů', 'ú'), ('y', 'ý'), ('z', 'ž')]
czech_diacritizables_chars = [char for sublist in czech_diacritics_tuples for char in sublist] + [char.upper() for sublist in
                                                                                                  czech_diacritics_tuples for char in
                                                                                                  sublist]

maltese_diacritics_tuples = [('c', 'ċ'), ('g', 'ġ'), ('h', 'ħ'), ('z', 'ż'), ('C', 'Ċ'), ('G', 'Ġ'), ('H', 'Ħ'), ('Z', 'Ż')]
maltese_diacritizable_chars = [char for sublist in maltese_diacritics_tuples for char in sublist] + [char.upper() for sublist in
                                                                                                  maltese_diacritics_tuples for char in
                                                                                                  sublist]
maltese_anglicization = {
    'ċ': 'c', 'Ċ': 'C',
    'ġ': 'g', 'Ġ': 'G', 
    'ħ': 'h', 'Ħ': 'H',
    'ż': 'z', 'Ż': 'Z'
}

def get_char_vocabulary(lang):
    if lang == 'mt':
        maltese_chars = 'ċġħż'
        maltese_chars_upper = maltese_chars.upper()
        allowed_chars = ', .'
        allowed_chars += string.ascii_lowercase + string.ascii_uppercase + maltese_chars + maltese_chars_upper
        return list(allowed_chars)
    elif lang == 'cs':
        czech_chars_with_diacritics = 'áčďěéíňóšřťůúýž'
        czech_chars_with_diacritics_upper = czech_chars_with_diacritics.upper()
        allowed_chars = ', .'
        allowed_chars += string.ascii_lowercase + string.ascii_uppercase + czech_chars_with_diacritics + czech_chars_with_diacritics_upper
        return list(allowed_chars)
    elif lang == 'en':
        allowed_chars = ', .'
        allowed_chars += string.ascii_lowercase + string.ascii_uppercase
        return list(allowed_chars)
    elif lang == 'de':
        german_special = 'ÄäÖöÜüẞß'
        allowed_chars = ', .'
        allowed_chars += string.ascii_lowercase + string.ascii_uppercase + german_special
        return list(allowed_chars)
    elif lang == 'ru':
        russian_special = 'бвгджзклмнпрстфхцчшщаэыуояеёюий'
        russian_special += russian_special.upper()
        russian_special += 'ЬьЪъ'
        allowed_chars = ', .'
        allowed_chars += russian_special
        return list(allowed_chars)


def get_token_vocabulary(tsv_token_file):
    tokens = []
    with open(tsv_token_file) as reader:
        for line in reader:
            line = line.strip('\n')
            token, freq = line.split('\t')

            if token.isalpha():
                tokens.append(token)

    return tokens

def introduce_token_level_errors_on_sentence(tokens, replace_prob, insert_prob, delete_prob, swap_prob, recase_prob, err_prob, std_dev,
                                             word_vocabulary, aspell_speller):
    """Basic token-level error operations"""
    num_errors = int(np.round(np.random.normal(err_prob, std_dev) * len(tokens)))
    num_errors = min(max(0, num_errors), len(tokens))  # num_errors \in [0; len(tokens)]

    if num_errors == 0:
        return ' '.join(tokens)
    token_ids_to_modify = np.random.choice(len(tokens), num_errors, replace=False)

    new_sentence = ''
    for token_id in range(len(tokens)):
        if token_id not in token_ids_to_modify:
            if new_sentence:
                new_sentence += ' '
            new_sentence += tokens[token_id]
            continue

        current_token = tokens[token_id]
        operation = np.random.choice(['replace', 'insert', 'delete', 'swap', 'recase'], p=[replace_prob, insert_prob, delete_prob,
                                                                                           swap_prob, recase_prob])
        new_token = ''
        if operation == 'replace':
            if not current_token.isalpha():
                new_token = current_token
            else:
                proposals = aspell_speller.suggest(current_token)[:10]
                if len(proposals) > 0:
                    new_token = np.random.choice(proposals)
                else:
                    new_token = current_token
        elif operation == 'insert':
            new_token = current_token + ' ' + np.random.choice(word_vocabulary)
        elif operation == 'delete':
            if not current_token.isalpha() or current_token in allowed_source_delete_tokens:
                new_token = current_token
            else:
                new_token = ''
        elif operation == 'recase':
            if not current_token.isalpha():
                new_token = current_token
            elif current_token.islower():
                new_token = current_token[0].upper() + current_token[1:]
            else:
                # either whole word is upper-case or mixed-case
                if np.random.random() < 0.5:
                    new_token = current_token.lower()
                else:
                    num_recase = min(len(current_token), max(1, int(np.round(np.random.normal(0.3, 0.4) * len(current_token)))))
                    char_ids_to_recase = np.random.choice(len(current_token), num_recase, replace=False)
                    new_token = ''
                    for char_i, char in enumerate(current_token):
                        if char_i in char_ids_to_recase:
                            if char.isupper():
                                new_token += char.lower()
                            else:
                                new_token += char.upper()
                        else:
                            new_token += char

        elif operation == 'swap':
            if token_id == len(tokens) - 1:
                continue

            new_token = tokens[token_id + 1]
            tokens[token_id + 1] = tokens[token_id]

        if new_sentence and new_token:
            new_sentence += ' '
        new_sentence = new_sentence + new_token

    return new_sentence

def introduce_extended_token_operations(tokens, replace_prob, insert_prob, delete_prob, swap_prob, unk_prob, 
                                      replace_random_prob, err_prob, std_dev, word_vocabulary, aspell_speller):
    """Extended token-level error operations for extended mode"""
    num_errors = int(np.round(np.random.normal(err_prob, std_dev) * len(tokens)))
    num_errors = min(max(0, num_errors), len(tokens))

    if num_errors == 0:
        return ' '.join(tokens)
    
    token_ids_to_modify = np.random.choice(len(tokens), num_errors, replace=False)

    new_sentence = ''
    for token_id in range(len(tokens)):
        if token_id not in token_ids_to_modify:
            if new_sentence:
                new_sentence += ' '
            new_sentence += tokens[token_id]
            continue

        current_token = tokens[token_id]
        operation = np.random.choice(['replace', 'insert', 'delete', 'swap', 'unk', 'replace_random'], 
                                   p=[replace_prob, insert_prob, delete_prob, swap_prob, unk_prob, replace_random_prob])
        
        new_token = ''
        if operation == 'replace':
            if not current_token.isalpha():
                new_token = current_token
            else:
                proposals = aspell_speller.suggest(current_token)[:10]
                if len(proposals) > 0:
                    new_token = np.random.choice(proposals)
                else:
                    new_token = current_token
        elif operation == 'insert':
            new_token = current_token + ' ' + np.random.choice(word_vocabulary)
        elif operation == 'delete':
            if not current_token.isalpha() or current_token in allowed_source_delete_tokens:
                new_token = current_token
            else:
                new_token = ''
        elif operation == 'swap':
            if token_id == len(tokens) - 1:
                new_token = current_token
            else:
                new_token = tokens[token_id + 1]
                tokens[token_id + 1] = tokens[token_id]
        elif operation == 'unk':
            new_token = '<UNK>'
        elif operation == 'replace_random':
            new_token = np.random.choice(word_vocabulary)

        if new_sentence and new_token:
            new_sentence += ' '
        new_sentence = new_sentence + new_token

    return new_sentence

def apply_sentence_level_operations(tokens, reverse_prob, mono_prob):
    """Apply sentence-level operations (reverse, mono) for extended mode"""
    new_tokens = tokens.copy()
    
    # Apply reverse with specified probability
    if np.random.random() < reverse_prob:
        new_tokens = new_tokens[::-1]
        return new_tokens
    
    # Apply mono with specified probability (mutually exclusive with reverse)
    if np.random.random() < mono_prob:
        # Keep 60% of tokens in original positions
        fraction_to_keep = 0.6
        num_to_keep = int(len(new_tokens) * fraction_to_keep)
        
        if num_to_keep < len(new_tokens):
            indices_to_keep = sorted(np.random.choice(len(new_tokens), num_to_keep, replace=False))
            
            # Extract tokens to keep and to reorder
            tokens_to_keep = [(idx, new_tokens[idx]) for idx in indices_to_keep]
            tokens_to_reorder = [new_tokens[i] for i in range(len(new_tokens)) if i not in indices_to_keep]
            
            # Shuffle the tokens to reorder
            np.random.shuffle(tokens_to_reorder)
            
            # Reconstruct the sentence
            result = [''] * len(new_tokens)
            for idx, token in tokens_to_keep:
                result[idx] = token
            
            # Fill in the remaining positions
            reorder_idx = 0
            for i in range(len(result)):
                if result[i] == '':
                    result[i] = tokens_to_reorder[reorder_idx]
                    reorder_idx += 1
            
            new_tokens = result
    
    return new_tokens


def introduce_char_level_errors_on_sentence(sentence, replace_prob, insert_prob, delete_prob, swap_prob, 
                                          change_diacritics_prob, anglicize_prob, err_prob, std_dev, char_vocabulary):
    sentence = list(sentence)
    num_errors = int(np.round(np.random.normal(err_prob, std_dev) * len(sentence)))
    num_errors = min(max(0, num_errors), len(sentence))

    if num_errors == 0:
        return ''.join(sentence)

    char_ids_to_modify = np.random.choice(len(sentence), num_errors, replace=False)

    new_sentence = ''
    for char_id in range(len(sentence)):
        if char_id not in char_ids_to_modify:
            new_sentence += sentence[char_id]
            continue

        operation = np.random.choice(['replace', 'insert', 'delete', 'swap', 'change_diacritics', 'anglicize'], 1,
                                     p=[replace_prob, insert_prob, delete_prob, swap_prob, change_diacritics_prob, anglicize_prob])

        current_char = sentence[char_id]
        new_char = ''
        if operation == 'replace':
            if current_char.isalpha():
                new_char = np.random.choice(char_vocabulary)
            else:
                new_char = current_char
        elif operation == 'insert':
            new_char = current_char + np.random.choice(char_vocabulary)
        elif operation == 'delete':
            if current_char.isalpha():
                new_char = ''
            else:
                new_char = current_char
        elif operation == 'swap':
            if char_id == len(sentence) - 1:
                new_char = current_char
            else:
                new_char = sentence[char_id + 1]
                sentence[char_id + 1] = sentence[char_id]
        elif operation == 'change_diacritics':
            if current_char in maltese_diacritizable_chars:
                for group in maltese_diacritics_tuples:
                    if current_char in group:
                        other_chars = [c for c in group if c != current_char]
                        if other_chars:
                            new_char = np.random.choice(other_chars)
                        else:
                            new_char = current_char
                        break
                else:
                    new_char = current_char
            else:
                new_char = current_char
        elif operation == 'anglicize':
            if current_char in maltese_anglicization:
                new_char = maltese_anglicization[current_char]
            else:
                new_char = current_char

        new_sentence += new_char

    return new_sentence

def load_asr_mapping(asr_mapping_file):
    """Load ASR error mapping from file"""
    asr_mapping = {}
    if os.path.exists(asr_mapping_file):
        with open(asr_mapping_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '\t' in line:
                    original, asr_transcription = line.split('\t', 1)
                    asr_mapping[original] = asr_transcription
    return asr_mapping

def apply_asr_errors(sentence, asr_mapping, asr_prob):
    """Apply ASR errors if sentence is in mapping"""
    if np.random.random() < asr_prob and sentence in asr_mapping:
        return asr_mapping[sentence]
    return sentence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("token_file", type=str, help="TSV file with tokens.")

    parser.add_argument("--lang", type=str, default="mt", help="Language identifier for ASpell (e.g. cs, en, de, ru).")
    parser.add_argument("--mode", type=str, default="basic", choices=["basic", "extended", "asr"], 
                       help="Error introduction mode.")
    
    # Basic token-level parameters
    parser.add_argument("--token_err_prob", default=0.15, type=float, help="Probability of token error.")
    parser.add_argument("--token_std_dev", default=0.2, type=float, help="Standard deviation of token error.")
    parser.add_argument("--token_err_distribution", default="0.7_0.1_0.1_0.1_0", type=str,
                        help="Basic mode error probabilities: replace_insert_delete_swap_recase")
    
    # Extended token-level parameters
    parser.add_argument("--extended_token_distribution", default="0.65_0.05_0.05_0.05_0.1_0.1", type=str,
                       help="Extended mode error probabilities: replace_insert_delete_swap_unk_replace_random")
    parser.add_argument("--reverse_prob", default=0.02, type=float, help="Probability of reverse operation.")
    parser.add_argument("--mono_prob", default=0.02, type=float, help="Probability of mono operation.")

    # Character-level parameters  
    parser.add_argument("--char_err_prob", default=0.02, type=float, help="Probability of character error.")
    parser.add_argument("--char_std_dev", default=0.01, type=float, help="Standard deviation of character error.")
    parser.add_argument("--char_err_distribution", default="0.2_0.2_0.2_0.2_0.2_0", type=str,
                       help="Character error probabilities: replace_insert_delete_swap_change_diacritics_anglicize")
    
    # ASR parameters
    parser.add_argument("--asr_mapping_file", type=str, default="", help="File containing ASR error mappings.")
    parser.add_argument("--asr_prob", default=0.04, type=float, help="Probability of ASR error substitution.")

    args = parser.parse_args()

    # Load vocabularies
    tokens = get_token_vocabulary(args.token_file)
    characters = get_char_vocabulary(args.lang)
    aspell_speller = aspell.Speller('lang', args.lang)
    
    # Parse error distributions based on mode
    if args.mode == "basic":
        token_err_distribution = args.token_err_distribution.split('_')
        if len(token_err_distribution) != 5:
            raise ValueError('Basic mode requires exactly 5 token error probabilities')
        token_replace_prob, token_insert_prob, token_delete_prob, token_swap_prob, recase_prob = map(float, token_err_distribution)
        if not np.isclose(sum(map(float, token_err_distribution)), 1.0):
            raise ValueError('Token error probabilities must sum to 1.0')
    
    elif args.mode == "extended":
        extended_distribution = args.extended_token_distribution.split('_')
        if len(extended_distribution) != 6:
            raise ValueError('Extended mode requires exactly 6 token error probabilities')
        if not np.isclose(sum(map(float, extended_distribution)), 1.0):
            raise ValueError('Extended token error probabilities must sum to 1.0')

    # Parse character error distribution
    char_err_distribution = args.char_err_distribution.split('_')
    if len(char_err_distribution) != 6:
        raise ValueError('Character error distribution requires exactly 6 values')
    char_replace_prob, char_insert_prob, char_delete_prob, char_swap_prob, change_diacritics_prob, anglicize_prob = map(float, char_err_distribution)
    if not np.isclose(sum(map(float, char_err_distribution)), 1.0):
        raise ValueError('Character error probabilities must sum to 1.0')

    # Load ASR mapping if needed
    asr_mapping = {}
    if args.mode == "asr" and args.asr_mapping_file:
        asr_mapping = load_asr_mapping(args.asr_mapping_file)
        print(f"Loaded {len(asr_mapping)} ASR mappings", file=sys.stderr)

    # Process input
    for line in fileinput.input(('-',)):
        input_line = line.strip('\n')
        if not input_line:
            continue
            
        # Apply ASR errors first if in ASR mode
        if args.mode == "asr":
            input_line = apply_asr_errors(input_line, asr_mapping, args.asr_prob)
        
        # Tokenize
        tokens_list = input_line.split(' ')
        
        if args.mode == "extended":
            # First apply sentence-level operations (reverse or mono)
            tokens_list = apply_sentence_level_operations(tokens_list, args.reverse_prob, args.mono_prob)
            
            # Then apply token-level errors with extended operations
            extended_probs = list(map(float, args.extended_token_distribution.split('_')))
            line_with_errors = introduce_extended_token_operations(
                tokens_list, extended_probs[0], extended_probs[1], extended_probs[2], 
                extended_probs[3], extended_probs[4], extended_probs[5],
                args.token_err_prob, args.token_std_dev, tokens, aspell_speller
            )
        else:
            # Apply basic token-level errors
            line_with_errors = introduce_token_level_errors_on_sentence(
                tokens_list, token_replace_prob, token_insert_prob, token_delete_prob,
                token_swap_prob, recase_prob, args.token_err_prob, args.token_std_dev, tokens, aspell_speller
            )
        
        # Apply character-level errors
        line_with_errors = introduce_char_level_errors_on_sentence(
            line_with_errors, char_replace_prob, char_insert_prob, char_delete_prob, 
            char_swap_prob, change_diacritics_prob, anglicize_prob, args.char_err_prob, 
            args.char_std_dev, characters
        )
        
        print(f'{input_line}\t{line_with_errors}')