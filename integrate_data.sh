#data integration scrips which combines rnaseq data with genomic data to create final dataset
# biological replicates are averaged out, so each tissue-sequence-5'utr combination appears only once
python src/integrate_sequence_expression.py \
  --rnaseq data/w1118_vst.tsv \
  --promoters data/sanitized_genome/promoters.fasta \
  --utr5-exons data/sanitized_genome/utr5_exons.fasta \
  --output data/dmel_integrated_data.tsv
