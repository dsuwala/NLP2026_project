#!/usr/bin/env python3
"""
Drosophila Melanogaster Synthetic RNA-seq Expression Data Generator

This script generates mock genomic datasets designed to stress-test sequence-to-scalar 
regression Transformers. It maps a tissue type context alongside a combined 
[Promoter + 5' UTR] sequence to a single continuous variance-stabilized (VST) 
expression value.

The output format is a tab-separated values (TSV) file containing a commented 
header block explaining the underlying biological rules used to generate the targets.
"""

import argparse
import random
import sys

# Define constants for biological simulation
TISSUES = [
    "BRAIN", "MIDGUT", "HINDGUT", "MALPIGHIAN_TUBULE", 
    "OVARY", "TESTIS", "FAT_BODY", "EYE"
]
NUCLEOTIDES = ["A", "C", "T", "G"]
# Real promoters are often AT-rich; we skew the nucleotide weights accordingly
NUCLEOTIDE_WEIGHTS = [0.35, 0.15, 0.35, 0.15] 

TARGET_MOTIF = "GATTACAA"

def parse_arguments():
    """
    Parses command-line arguments using argparse.
    
    Returns:
        argparse.Namespace: Object containing validated command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic promoter sequence and VST expression datasets for Drosophila ML tasks."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="synthetic_drosophila_data.tsv",
        help="Path to the output TSV file (default: synthetic_drosophila_data.tsv)"
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=2000,
        help="Total number of gene-tissue pairs to generate (default: 2000)"
    )
    parser.add_argument(
        "--promoter-len", "-p",
        type=int,
        default=1000,
        help="Fixed length of the promoter sequence in base pairs (default: 1000)"
    )
    parser.add_argument(
        "--min-utr-len",
        type=int,
        default=50,
        help="Minimum length of the 5' UTR sequence (default: 50)"
    )
    parser.add_argument(
        "--max-utr-len", "-u",
        type=int,
        default=250,
        help="Maximum length of the 5' UTR sequence (default: 250)"
    )
    parser.add_argument(
        "--motif-inject-rate", "-r",
        type=float,
        default=0.25,
        help="Fraction of BRAIN samples forced to contain the target motif (default: 0.25)"
    )
    return parser.parse_args()

def generate_random_dna(length):
    """
    Generates a random DNA string of a specified length based on skewed AT/CG weights.
    
    Args:
        length (int): Length of the desired DNA string.
        
    Returns:
        str: Generated DNA sequence.
    """
    return "".join(random.choices(NUCLEOTIDES, weights=NUCLEOTIDE_WEIGHTS, k=length))

def inject_motif(sequence, motif):
    """
    Randomly places a specific sequence motif inside an existing DNA sequence string,
    overwriting the original bases at that position.
    
    Args:
        sequence (str): The background DNA sequence.
        motif (str): The sequence motif to insert.
        
    Returns:
        str: The altered DNA sequence containing the motif.
    """
    if len(sequence) <= len(motif):
        return motif
    insert_idx = random.randint(0, len(sequence) - len(motif))
    return sequence[:insert_idx] + motif + sequence[insert_idx + len(motif):]

def calculate_vst_expression(tissue, sequence, motif):
    """
    Applies the ground-truth biological rules to determine the continuous 
    expression score (simulating Variance Stabilized Transformation counts).
    
    Rules:
        - Basal background expression is drawn from a continuous Gamma distribution (range ~0.0 to 6.0).
        - If tissue is 'BRAIN' and the sequence contains 'GATTACAA', expression spikes heavily (range ~12.0 to 16.0).
        - Minor continuous noise is added globally to simulate stochastic experimental variance.
        
    Args:
        tissue (str): The tissue context token.
        sequence (str): The complete concatenated DNA string.
        motif (str): The sequence motif driving the tissue-specific rule.
        
    Returns:
        float: Calculated VST expression level rounded to 4 decimal places.
    """
    # Generate background basal expression using a gamma distribution
    basal_expression = random.gammavariate(alpha=2.0, beta=1.5)
    
    # Check for the causal interaction (Happy Path verification rule)
    if tissue == "BRAIN" and (motif in sequence):
        # High upregulation override simulating a strong tissue-specific transcription factor activation
        specific_activation = random.uniform(11.0, 13.0)
        final_score = specific_activation + random.normalvariate(0.0, 0.5)
    else:
        # Standard tissue distribution with occasional background fluctuations
        final_score = basal_expression + random.normalvariate(0.0, 0.2)
        
    # Clamp to ensure no negative expressions, mimicking VST behavior
    return round(max(0.0, final_score), 4)

def main():
    args = parse_arguments()
    
    print(f"Generating {args.num_samples} synthetic genomic data samples...")
    print(f"Target file: {args.output}")
    
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            # Write the required Ground Truth metadata explanation block at the absolute top
            f.write("# ====================================================================\n")
            f.write("# GENOMIC TRANSFORMER PIPELINE SANITY CHECK DATASET\n")
            f.write("# ====================================================================\n")
            f.write("# GROUND TRUTH RULES BAKED INTO THIS DATASET:\n")
            f.write("# 1. Continuous targets mimic Variance Stabilized Transformation (VST) values (Range: 0.0 to 18.0).\n")
            f.write("# 2. Basal/Housekeeping expressions follow a standard continuous low-level distribution.\n")
            f.write("# 3. CAUSAL LOGIC CIRCUIT: If TISSUE == 'BRAIN' AND the complete sequence contains the\n")
            f.write(f"#    transcription factor binding motif '{TARGET_MOTIF}', the output VST target is heavily\n")
            f.write("#    upregulated into a high-expression state (11.0 - 14.5).\n")
            f.write("# 4. Use this embedded circuit to test if your Attention Heads or Activation Patching\n")
            f.write("#    can successfully back-trace and isolate the motif syntax.\n")
            f.write("# ====================================================================\n")
            
            # Write file columns
            f.write("gene_id\ttissue\tpromoter_sequence\tutr_5_sequence\tvst_expression\n")
            
            for i in range(args.num_samples):
                gene_id = f"FBgn{i+1:07d}" # Simulates standard FlyBase gene naming format
                tissue = random.choice(TISSUES)
                
                # Construct sequence inputs
                promoter = generate_random_dna(args.promoter_len)
                utr_len = random.randint(args.min_utr_len, args.max_utr_len)
                utr_5 = generate_random_dna(utr_len)
                
                # Conditionally plant the motif to make sure the causal graph has enough positive examples
                if tissue == "BRAIN" and random.random() < args.motif_inject_rate:
                    # Randomly inject into either the promoter or the UTR region
                    if random.choice([True, False]):
                        promoter = inject_motif(promoter, TARGET_MOTIF)
                    else:
                        utr_5 = inject_motif(utr_5, TARGET_MOTIF)
                
                # Combine sequences to evaluate true causal presence
                combined_sequence = promoter + utr_5
                vst_target = calculate_vst_expression(tissue, combined_sequence, TARGET_MOTIF)
                
                # Write record
                f.write(f"{gene_id}\t{tissue}\t{promoter}\t{utr_5}\t{vst_target}\n")
                
        print(f"Data successfully compiled! Review data shape and headers in '{args.output}'.")
        
    except IOError as e:
        print(f"File writing error encountered: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
