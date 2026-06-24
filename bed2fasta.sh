#script to convert bed files which contains adresses to relevant genes and 5'UTR fragments 
#to fasta with that sequence retrieved
#promoters
bedtools getfasta \
    -fi data/genome/dmel-all-chromosome-r6.16.fasta \
    -bed data/sanitized_genome/promoters.bed \
    -fo data/sanitized_genome/promoters.fasta \
    -name \
    -s

# utr
bedtools getfasta \
    -fi data/genome/dmel-all-chromosome-r6.16.fasta \
    -bed data/sanitized_genome/utr5_exons.bed \
    -fo data/sanitized_genome/utr5_exons.fasta \
    -name \
    -s