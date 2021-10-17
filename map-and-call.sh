#!/bin/bash

REF_12X="GCF_000003745.3_12X_genomic.fna"
REF_URL="https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/003/745/GCF_000003745.3_12X/GCF_000003745.3_12X_genomic.fna.gz"
REGIONS="527.regions.txt"
R1=$1
R2=$2
PREFIX=${R1%f*q.gz}
THREADS=4 # Bowtie2 and samtools sort threads
MEM="4G" # Max memory per thread for samtools sort

#Download reference and create index if not exist
if [[ ! -f  $REF_12X ]]; then
	wget $REF_URL
	which pigz && pigz -d ${REF_12X}.gz || gunzip ${REF_12X}.gz
fi

if [[ ! -f  12x_index.1.bt2 ]]; then
	bowtie2-build --threads $THREADS -q $REF_12X.head 12x_index && echo "Reference indexed"
fi

#Map reads, process mapping file, mpileup and call SNPs
bowtie2 -x 12x_index -1 $R1 -2 $R2 -p $THREADS 2>$PREFIX.bowtie.err | \
samtools view -u -F 4 - | \
samtools sort -@ $THREADS -m 4G -o $PREFIX.bam

bcftools mpileup -f $REF_12X -Q 18 --regions-file $REGIONS $PREFIX.bam --max-depth 500 -O u -o $PREFIX.mpileup.bcf $PREFIX.bam 
bcftools call -O v -o $PREFIX.calls.vcf -V indels -m $PREFIX.mpileup.bcf
bcftools query -f '%CHROM %POS  %REF  %ALT [ %TGT]\n' $PREFIX.calls.vcf > $PREFIX.GT.tsv
