# Source code and data for Russian grape parentage study

This repository contains source code and data relating to "SNP-based analysis reveals authenticity and genetic similarity of Russian indigenous grape cultivars" article.

## Source code

### genotype-scaling.py

The script performs multidimensinal scaling with tSNE [implementation](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) from [scikit-learn](https://scikit-learn.org/stable/index.html) or UMAP (Uniform Manifold Approximation and Projection) https://umap-learn.readthedocs.io/en/latest/. It requires genotypes in tab-separated table and metadata, including 1 or 2 user-defined categories showing as colors and different characters on the plots.

### genotype-analysis.py

This script implements all-in-one analysis: IBS-distance calculation, Opposing homozygotes counting, finding of possible parent-offspring relations and parentage trios, plotting dendrograms based on IBS-distance and 2D-representation of IBS-distance data using tSNE.

### Requirements

* numpy
* pandas
* umap
* matplotlib
* seaborn
* sklearn
* scipy

The packages may be easilly installed with pip or conda.

#### Usage

-h, --help  
show the help message and exit

-i INPUT, --input INPUT  
Genotype in tab-separated table.

-m METADATA, --metadata METADATA  
Tab-separated metadata for genotypes, if available. First column contains identifiers and should match genotype IDs from first row of input datafile.

-s SHIFT, --shift SHIFT  
Number of columns between first column and columns with genotypes.

-a {umap,tsne}, --algo {umap,tsne}
Algorithm for dimensionality reduction: umap (default) or tsne.

-p PERPLEXITY, --perplexity PERPLEXITY  
tSNE perplexity parameter.


## Data and metadata

### snp.data.tsv and metadata.tsv

Metadata (geographical group and berry colour) and genotypes of studied V. vinifiera cultivars coupled with data from Laucou et al., 2018.
Both tables are tab-separated.

### References

McInnes, L., Healy, J., & Melville, J. (2018). Umap: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426.

Van Der Maaten, L. (2014). Accelerating t-SNE using tree-based algorithms. The Journal of Machine Learning Research, 15(1), 3221-3245.

Laucou, V. et al. (2018). Extended diversity analysis of cultivated grapevine Vitis vinifera with 10K genome-wide SNPs. PloS one, 13(2), e0192540.
