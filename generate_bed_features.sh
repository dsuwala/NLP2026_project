# cleaning stage to preprocess genonme and its annotation unifying names and standardizing adressing protocol
#produces bed files (promoters.bed and utr5_exons.bed) which contains unified adresses along the genome
python src/generate_bed_features.py \
  --fasta data/genome/dmel-all-chromosome-r6.16.fasta \
  --gff data/genome/dmel-all-no-analysis-r6.16.gff \
  --output-dir data/bed_outputs