#!/usr/bin/env python3
"""
evaluate_all.py - Complete evaluation pipeline for Maltese GEC
Includes detokenization, M2 scoring, ERRANT, and BLEU
"""

import os
import sys
import json
import argparse
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Tuple
from difflib import SequenceMatcher

# Try to import tokenizer
try:
    from tokenisation import MTWordTokenizer
    MT_TOKENIZER_AVAILABLE = True
    print("Successfully imported MTWordTokenizer from tokenisation.py")
except ImportError as e:
    print(f"First import attempt failed: {e}")
    try:
        # Try with explicit path
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from tokenisation import MTWordTokenizer
        MT_TOKENIZER_AVAILABLE = True
        print("Successfully imported MTWordTokenizer with path adjustment")
    except ImportError as e2:
        print(f"Second import attempt failed: {e2}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
        print(f"Files in script directory: {os.listdir(os.path.dirname(os.path.abspath(__file__)))}")
        MT_TOKENIZER_AVAILABLE = False

# Try to import sacrebleu
try:
    from sacrebleu import corpus_bleu
    SACREBLEU_AVAILABLE = True
except ImportError:
    print("Warning: sacrebleu not installed. BLEU scoring will be skipped.")
    print("Install with: pip install sacrebleu")
    SACREBLEU_AVAILABLE = False


class Detokenizer:
    """Handle detokenization with Maltese-specific rules"""
    
    def __init__(self, use_mt_tokenizer=True):
        self.tokenizer = None
        if use_mt_tokenizer and MT_TOKENIZER_AVAILABLE:
            try:
                self.tokenizer = MTWordTokenizer()
                print("Using Maltese tokenizer for detokenization")
            except Exception as e:
                print(f"Failed to initialize MT tokenizer: {e}")
        
        if self.tokenizer is None:
            print("Using simple rule-based detokenization")
    
    def detokenize_line(self, line: str) -> str:
        """Detokenize a single line"""
        if self.tokenizer is not None:
            try:
                tokens = line.strip().split()
                return self.tokenizer.detokenize(tokens)
            except Exception as e:
                print(f"MT detokenization failed: {e}")
        
        # Fallback to simple rules
        # Handle punctuation
        line = re.sub(r'\s+([.,!?;:\'"])', r'\1', line)
        # Handle quotes
        line = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', line)
        line = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", line)
        # Handle Maltese apostrophes
        line = re.sub(r"\s+([''])", r'\1', line)
        line = re.sub(r"([''])\s+", r'\1', line)
        # Clean multiple spaces
        line = re.sub(r'\s+', ' ', line)
        return line.strip()
    
    def detokenize_file(self, input_file: str, output_file: str):
        """Detokenize entire file"""
        with open(input_file, 'r', encoding='utf-8') as fin, \
             open(output_file, 'w', encoding='utf-8') as fout:
            for line in fin:
                detok_line = self.detokenize_line(line)
                fout.write(detok_line + '\n')

class TokenSpanMetrics:
    """Calculate token-level and span-level detection and correction metrics for GEC"""
    
    def calculate_all_metrics(self, source_file: str, hypothesis_file: str, reference_file: str) -> Dict:
        """Calculate all detection and correction metrics
        
        Args:
            source_file: Path to source sentences (one per line)
            hypothesis_file: Path to system predictions (one per line)
            reference_file: Path to gold corrections (one per line)
            
        Returns:
            Dictionary containing token and span metrics for detection and correction
        """
        results = {
            'token_detection': self._calculate_token_detection(source_file, hypothesis_file, reference_file),
            'token_correction': self._calculate_token_correction(source_file, hypothesis_file, reference_file),
            'span_detection': self._calculate_span_detection(source_file, hypothesis_file, reference_file),
            'span_correction': self._calculate_span_correction(source_file, hypothesis_file, reference_file)
        }
        return results
    
    def _calculate_token_detection(self, source_file: str, hypothesis_file: str, reference_file: str) -> Dict:
        """Token-level error detection metrics"""
        tp = fp = fn = 0
        
        with open(source_file, 'r', encoding='utf-8') as sf, \
             open(hypothesis_file, 'r', encoding='utf-8') as hf, \
             open(reference_file, 'r', encoding='utf-8') as rf:
            
            for src_line, hyp_line, ref_line in zip(sf, hf, rf):
                src_tokens = src_line.strip().split()
                hyp_tokens = hyp_line.strip().split()
                ref_tokens = ref_line.strip().split()
                
                # Align source with reference and hypothesis
                src_errors = set()  # Token positions with errors
                ref_matcher = SequenceMatcher(None, src_tokens, ref_tokens)
                
                for tag, i1, i2, j1, j2 in ref_matcher.get_opcodes():
                    if tag != 'equal':
                        src_errors.update(range(i1, i2))
                
                # Check hypothesis detections
                hyp_detections = set()
                hyp_matcher = SequenceMatcher(None, src_tokens, hyp_tokens)
                
                for tag, i1, i2, j1, j2 in hyp_matcher.get_opcodes():
                    if tag != 'equal':
                        hyp_detections.update(range(i1, i2))
                
                # Calculate metrics
                tp += len(src_errors & hyp_detections)
                fp += len(hyp_detections - src_errors)
                fn += len(src_errors - hyp_detections)
        
        return self._calculate_metrics(tp, fp, fn)
    
    def _calculate_token_correction(self, source_file: str, hypothesis_file: str, reference_file: str) -> Dict:
        """Token-level error correction metrics"""
        tp = fp = fn = 0
        
        with open(source_file, 'r', encoding='utf-8') as sf, \
             open(hypothesis_file, 'r', encoding='utf-8') as hf, \
             open(reference_file, 'r', encoding='utf-8') as rf:
            
            for src_line, hyp_line, ref_line in zip(sf, hf, rf):
                hyp_tokens = hyp_line.strip().split()
                ref_tokens = ref_line.strip().split()
                
                # Direct token-by-token comparison
                matcher = SequenceMatcher(None, hyp_tokens, ref_tokens)
                
                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    if tag == 'equal':
                        tp += (i2 - i1)
                    elif tag == 'replace':
                        # Check if any tokens match
                        hyp_span = hyp_tokens[i1:i2]
                        ref_span = ref_tokens[j1:j2]
                        matches = 0
                        for h_tok in hyp_span:
                            if h_tok in ref_span:
                                matches += 1
                                ref_span.remove(h_tok)  # Remove to avoid double counting
                        tp += matches
                        fp += len(hyp_span) - matches
                        fn += len(ref_tokens[j1:j2]) - matches
                    elif tag == 'delete':
                        fp += (i2 - i1)
                    elif tag == 'insert':
                        fn += (j2 - j1)
        
        return self._calculate_metrics(tp, fp, fn)
    
    def _calculate_span_detection(self, source_file: str, hypothesis_file: str, reference_file: str) -> Dict:
        """Span-level error detection metrics"""
        tp = fp = fn = 0
        
        with open(source_file, 'r', encoding='utf-8') as sf, \
             open(hypothesis_file, 'r', encoding='utf-8') as hf, \
             open(reference_file, 'r', encoding='utf-8') as rf:
            
            for src_line, hyp_line, ref_line in zip(sf, hf, rf):
                src_tokens = src_line.strip().split()
                hyp_tokens = hyp_line.strip().split()
                ref_tokens = ref_line.strip().split()
                
                # Get error spans in source
                src_spans = []
                ref_matcher = SequenceMatcher(None, src_tokens, ref_tokens)
                for tag, i1, i2, j1, j2 in ref_matcher.get_opcodes():
                    if tag != 'equal':
                        src_spans.append((i1, i2))
                
                # Get detected spans in hypothesis
                hyp_spans = []
                hyp_matcher = SequenceMatcher(None, src_tokens, hyp_tokens)
                for tag, i1, i2, j1, j2 in hyp_matcher.get_opcodes():
                    if tag != 'equal':
                        hyp_spans.append((i1, i2))
                
                # Match spans
                matched_src = set()
                matched_hyp = set()
                
                for i, src_span in enumerate(src_spans):
                    for j, hyp_span in enumerate(hyp_spans):
                        if self._spans_overlap(src_span, hyp_span):
                            matched_src.add(i)
                            matched_hyp.add(j)
                
                tp += len(matched_src)
                fp += len(hyp_spans) - len(matched_hyp)
                fn += len(src_spans) - len(matched_src)
        
        return self._calculate_metrics(tp, fp, fn)
    
    def _calculate_span_correction(self, source_file: str, hypothesis_file: str, reference_file: str) -> Dict:
        """Span-level error correction metrics"""
        tp = fp = fn = 0
        
        with open(source_file, 'r', encoding='utf-8') as sf, \
             open(hypothesis_file, 'r', encoding='utf-8') as hf, \
             open(reference_file, 'r', encoding='utf-8') as rf:
            
            for src_line, hyp_line, ref_line in zip(sf, hf, rf):
                src_tokens = src_line.strip().split()
                hyp_tokens = hyp_line.strip().split()
                ref_tokens = ref_line.strip().split()
                
                # Get edits from source to reference
                ref_edits = []
                ref_matcher = SequenceMatcher(None, src_tokens, ref_tokens)
                for tag, i1, i2, j1, j2 in ref_matcher.get_opcodes():
                    if tag != 'equal':
                        ref_edits.append({
                            'src_span': (i1, i2),
                            'correction': ref_tokens[j1:j2],
                            'type': tag
                        })
                
                # Get edits from source to hypothesis
                hyp_edits = []
                hyp_matcher = SequenceMatcher(None, src_tokens, hyp_tokens)
                for tag, i1, i2, j1, j2 in hyp_matcher.get_opcodes():
                    if tag != 'equal':
                        hyp_edits.append({
                            'src_span': (i1, i2),
                            'correction': hyp_tokens[j1:j2],
                            'type': tag
                        })
                
                # Match edits
                matched_ref = set()
                matched_hyp = set()
                
                for i, ref_edit in enumerate(ref_edits):
                    for j, hyp_edit in enumerate(hyp_edits):
                        if (self._spans_overlap(ref_edit['src_span'], hyp_edit['src_span']) and
                            ref_edit['correction'] == hyp_edit['correction']):
                            matched_ref.add(i)
                            matched_hyp.add(j)
                
                tp += len(matched_ref)
                fp += len(hyp_edits) - len(matched_hyp)
                fn += len(ref_edits) - len(matched_ref)
        
        return self._calculate_metrics(tp, fp, fn)
    
    def _spans_overlap(self, span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
        """Check if two spans overlap"""
        return not (span1[1] <= span2[0] or span2[1] <= span1[0])
    
    def _calculate_metrics(self, tp: int, fp: int, fn: int) -> Dict:
        """Calculate precision, recall, and F0.5 from counts"""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f05 = (1.25 * precision * recall) / (0.25 * precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f0.5': f05
        }

class M2Evaluator:
    """Handle M2 format generation and scoring"""
    
    def __init__(self, m2scorer_path: str):
        self.m2scorer_path = m2scorer_path
        self.m2scorer_script = os.path.join(m2scorer_path, "scripts", "m2scorer.py")
        
        if not os.path.exists(self.m2scorer_script):
            raise FileNotFoundError(f"M2 scorer not found at {self.m2scorer_script}")
    
    def generate_m2_file(self, source_file: str, target_file: str, output_file: str):
        """Generate M2 format from parallel files"""
        with open(source_file, 'r', encoding='utf-8') as sf, \
             open(target_file, 'r', encoding='utf-8') as tf, \
             open(output_file, 'w', encoding='utf-8') as of:
            
            for src_line, tgt_line in zip(sf, tf):
                src_tokens = src_line.strip().split()
                tgt_tokens = tgt_line.strip().split()
                
                # Write source
                of.write('S ' + ' '.join(src_tokens) + '\n')
                
                # Get edits
                edits = self._get_edits(src_tokens, tgt_tokens)
                for edit in edits:
                    of.write(edit + '\n')
                
                of.write('\n')
    
    def _get_edits(self, source: List[str], target: List[str]) -> List[str]:
        """Extract edits between source and target"""
        edits = []
        matcher = SequenceMatcher(None, source, target)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                continue
            elif tag == 'replace':
                correction = ' '.join(target[j1:j2])
                edits.append(f"A {i1} {i2}|||UNK|||{correction}|||REQUIRED|||-NONE-|||0")
            elif tag == 'delete':
                edits.append(f"A {i1} {i2}|||UNK||||||REQUIRED|||-NONE-|||0")
            elif tag == 'insert':
                correction = ' '.join(target[j1:j2])
                edits.append(f"A {i1} {i1}|||UNK|||{correction}|||REQUIRED|||-NONE-|||0")
        
        if not edits:
            edits.append("A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0")
        
        return edits
    
    def run_m2_scorer(self, system_file: str, gold_m2: str) -> Dict:
        """Run official M2 scorer
        
        Args:
            system_file: Plain text file with predictions (one per line)
            gold_m2: M2 format file with source sentences and gold edits
        """
        try:
            # Verify files exist
            if not os.path.exists(system_file):
                print(f"System file not found: {system_file}")
                return None
            if not os.path.exists(gold_m2):
                print(f"Gold M2 file not found: {gold_m2}")
                return None
            
            cmd = [
                sys.executable,
                self.m2scorer_script,
                system_file,  # Plain text predictions
                gold_m2       # M2 format gold standard
            ]
            
            # Use older subprocess API for compatibility
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                print(f"M2 scorer error: {stderr.decode('utf-8') if stderr else 'Unknown error'}")
                return None
            
            # Parse output
            scores = {}
            output_text = stdout.decode('utf-8') if isinstance(stdout, bytes) else stdout
            for line in output_text.strip().split('\n'):
                if line.startswith("Precision"):
                    scores['precision'] = float(line.split()[-1])
                elif line.startswith("Recall"):
                    scores['recall'] = float(line.split()[-1])
                elif line.startswith("F"):
                    # Handle both F_0.5 and F0.5 formats
                    parts = line.split()
                    scores['f0.5'] = float(parts[-1])
            
            return scores
            
        except Exception as e:
            print(f"M2 scorer exception: {e}")
            import traceback
            traceback.print_exc()
            return None


class DebattistaERRANT:
    """Debattista ERRANT scoring"""
    
    def calculate_scores(self, source_file: str, hypothesis_file: str, reference_file: str) -> Dict:
        """Calculate ERRANT scores"""
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        with open(source_file, 'r', encoding='utf-8') as sf, \
             open(hypothesis_file, 'r', encoding='utf-8') as hf, \
             open(reference_file, 'r', encoding='utf-8') as rf:
            
            for src, hyp, ref in zip(sf, hf, rf):
                src_tokens = src.strip().split()
                hyp_tokens = hyp.strip().split()
                ref_tokens = ref.strip().split()
                
                # Align hypothesis with reference
                matcher = SequenceMatcher(None, hyp_tokens, ref_tokens)
                
                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    if tag == 'equal':
                        total_tp += (i2 - i1)
                    elif tag == 'replace':
                        num_tokens = max(i2 - i1, j2 - j1)
                        total_fp += num_tokens
                        total_fn += num_tokens
                    elif tag == 'delete':
                        total_fp += (i2 - i1)
                    elif tag == 'insert':
                        total_fn += (j2 - j1)
        
        # Calculate metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f05 = (1.25 * precision * recall) / (0.25 * precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
            'precision': precision,
            'recall': recall,
            'f0.5': f05
        }


def main():
    parser = argparse.ArgumentParser(description='Complete evaluation for Maltese GEC')
    parser.add_argument('predictions_file', help='Path to predictions file')
    parser.add_argument('--source', help='Source file (default: data/authentic/splits/test/src.txt)')
    parser.add_argument('--reference', help='Reference file (default: data/authentic/splits/test/trg.txt)')
    parser.add_argument('--no-detokenize', action='store_true', help='Skip detokenization')
    parser.add_argument('--output-dir', help='Output directory (default: same as predictions)')
    
    args = parser.parse_args()
    
    # Set defaults - detect base directory
    predictions_path = Path(args.predictions_file).resolve()
    
    # Find base directory by looking for malteseGEC in path
    base_dir = None
    for parent in predictions_path.parents:
        if parent.name == 'malteseGEC':
            base_dir = parent
            break
    
    if base_dir is None:
        # Fallback to common locations
        if Path("/home/fed/malteseGEC").exists():
            base_dir = Path("/home/fed/malteseGEC")
        elif Path("/app/malteseGEC").exists():
            base_dir = Path("/app/malteseGEC")
        elif Path("/root/malteseGEC").exists():
            base_dir = Path("/root/malteseGEC")
        else:
            base_dir = Path.cwd()
    
    if not args.source:
        args.source = str(base_dir / "data" / "authentic" / "splits" / "test" / "src.txt")
    if not args.reference:
        args.reference = str(base_dir / "data" / "authentic" / "splits" / "test" / "trg.txt")
    if not args.output_dir:
        args.output_dir = str(predictions_path.parent)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("MALTESE GEC EVALUATION")
    print("=" * 60)
    print(f"Base directory: {base_dir}")
    print(f"Predictions: {predictions_path}")
    print(f"Source: {args.source}")
    print(f"Reference: {args.reference}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Step 1: Detokenization
    if args.no_detokenize:
        print("Skipping detokenization (--no-detokenize flag)")
        clean_predictions = str(predictions_path)
        clean_source = args.source
        clean_reference = args.reference
    else:
        print("Step 1: Detokenizing files...")
        detokenizer = Detokenizer(use_mt_tokenizer=True)
        
        # Detokenize predictions
        clean_predictions = str(output_dir / "predictions_detok.txt")
        print(f"  Detokenizing predictions...")
        detokenizer.detokenize_file(str(predictions_path), clean_predictions)
        
        # Detokenize source and reference if needed
        clean_source = str(output_dir / "source_detok.txt")
        clean_reference = str(output_dir / "reference_detok.txt")
        
        print(f"  Detokenizing source...")
        detokenizer.detokenize_file(args.source, clean_source)
        
        print(f"  Detokenizing reference...")
        detokenizer.detokenize_file(args.reference, clean_reference)
    
    # Initialize results
    results = {
        'files': {
            'predictions': str(predictions_path),
            'source': args.source,
            'reference': args.reference
        }
    }
    
    # Step 2: M2 Evaluation
    print("\nStep 2: M2 Evaluation...")
    m2scorer_path = base_dir / "eval" / "m2scorer"
    
    if m2scorer_path.exists():
        try:
            m2_evaluator = M2Evaluator(str(m2scorer_path))
            
            # Generate M2 format for gold standard only
            gold_m2 = str(output_dir / "gold.m2")
            
            print("  Generating M2 format for gold standard...")
            m2_evaluator.generate_m2_file(clean_source, clean_reference, gold_m2)
            
            # M2 scorer expects: plain text predictions + M2 format gold
            print("  Running M2 scorer...")
            # Pass the clean predictions (plain text) and gold.m2
            m2_scores = m2_evaluator.run_m2_scorer(clean_predictions, gold_m2)
            
            if m2_scores:
                results['m2_official'] = m2_scores
                print(f"  M2 Precision: {m2_scores.get('precision', 'N/A'):.4f}")
                print(f"  M2 Recall: {m2_scores.get('recall', 'N/A'):.4f}")
                print(f"  M2 F0.5: {m2_scores.get('f0.5', 'N/A'):.4f}")
            else:
                print("  M2 scoring failed")
                results['m2_official'] = {'error': 'M2 scoring failed'}
                
        except Exception as e:
            print(f"  M2 evaluation error: {e}")
            results['m2_official'] = {'error': str(e)}
    else:
        print("  M2 scorer not found - skipping")
        results['m2_official'] = {'note': 'M2 scorer not installed'}
    
    # Step 3: Debattista ERRANT
    print("\nStep 3: Debattista-style ERRANT Evaluation...")
    errant = DebattistaERRANT()
    errant_scores = errant.calculate_scores(clean_source, clean_predictions, clean_reference)
    results['errant_debattista'] = errant_scores
    print(f"  ERRANT F0.5: {errant_scores['f0.5']:.4f}")
    
    # Step 4: BLEU
    if SACREBLEU_AVAILABLE:
        print("\nStep 4: BLEU Evaluation...")
        with open(clean_predictions, 'r', encoding='utf-8') as f:
            hypotheses = [line.strip() for line in f]
        with open(clean_reference, 'r', encoding='utf-8') as f:
            references = [line.strip() for line in f]
        
        bleu = corpus_bleu(hypotheses, [references])
        results['bleu'] = {
            'score': bleu.score,
            'bp': bleu.bp,
            'ratio': bleu.sys_len / bleu.ref_len if bleu.ref_len > 0 else 0,
            'hyp_len': bleu.sys_len,
            'ref_len': bleu.ref_len
        }
        print(f"  BLEU: {bleu.score:.2f}")
    else:
        print("\nStep 4: BLEU Evaluation skipped (sacrebleu not installed)")
        results['bleu'] = {'note': 'sacrebleu not installed'}
    
    # Step 5: Detection and Correction Metrics
    print("\nStep 5: Token/Span Detection and Correction Metrics...")
    try:
        detection = TokenSpanMetrics()
        detection_results = detection.calculate_all_metrics(clean_source, clean_predictions, clean_reference)
        results['detection_metrics'] = detection_results
        
        print(f"  Token-based Detection F0.5:   {detection_results['token_detection']['f0.5']:.4f}")
        print(f"  Token-based Correction F0.5:  {detection_results['token_correction']['f0.5']:.4f}")
        print(f"  Span-based Detection F0.5:    {detection_results['span_detection']['f0.5']:.4f}")
        print(f"  Span-based Correction F0.5:   {detection_results['span_correction']['f0.5']:.4f}")
    except Exception as e:
        print(f"  Detection metrics failed: {e}")
        results['detection_metrics'] = {'error': str(e)}
    
    # Save results
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Create comparison file
    print("\nStep 6: Creating comparison file...")
    comparison_file = output_dir / "comparison.tsv"
    with open(clean_source, 'r', encoding='utf-8') as sf, \
         open(clean_predictions, 'r', encoding='utf-8') as pf, \
         open(clean_reference, 'r', encoding='utf-8') as rf, \
         open(comparison_file, 'w', encoding='utf-8') as cf:
        
        cf.write("SOURCE\tPREDICTION\tREFERENCE\n")
        for src, pred, ref in zip(sf, pf, rf):
            cf.write(f"{src.strip()}\t{pred.strip()}\t{ref.strip()}\n")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    if 'f0.5' in results.get('m2_official', {}):
        print(f"M2 F0.5 (Official):            {results['m2_official']['f0.5']:.4f}")
    
    print(f"ERRANT F0.5 (Debattista):      {results['errant_debattista']['f0.5']:.4f}")
    
    if 'score' in results.get('bleu', {}):
        print(f"BLEU:                          {results['bleu']['score']:.2f}")
    
    print("\nDetection/Correction Metrics:")
    if 'detection_metrics' in results and 'error' not in results['detection_metrics']:
        detection_results = results['detection_metrics']
        print(f"  Token Detection F0.5:        {detection_results['token_detection']['f0.5']:.4f}")
        print(f"  Token Correction F0.5:       {detection_results['token_correction']['f0.5']:.4f}")
        print(f"  Span Detection F0.5:         {detection_results['span_detection']['f0.5']:.4f}")
        print(f"  Span Correction F0.5:        {detection_results['span_correction']['f0.5']:.4f}")
    else:
        print("  Detection metrics not available")
    
    print("\nFiles created:")
    print(f"  - {results_file}")
    print(f"  - {comparison_file}")
    if not args.no_detokenize:
        print(f"  - {clean_predictions} (detokenized)")
    print(f"  - {output_dir}/gold.m2")
    print(f"  - {output_dir}/system.m2")
    
    print("\nDone!")


if __name__ == '__main__':
    main()