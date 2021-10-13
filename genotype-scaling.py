#!/usr/bin/env python
import argparse
import numpy as np
import os
import pandas
import sys
import umap
from matplotlib import pyplot
from seaborn import relplot
from seaborn import color_palette
from sklearn.manifold import TSNE
from scipy.spatial.distance import squareform


def parse_args():
	'''Parse command-line arguments'''
	parser = argparse.ArgumentParser(
		description="Multidimensinal genotype scaling with tSNE and UMAP. "
		+ "Input files are tab-separated tables. " 
		+ "Ouptut files are images in PNG and SVG formats and " 
		+ "table with coordinates in lower dimensional space.")
	parser.add_argument("-i", "--input", required=True,
		help="Genotype in tab-separated table.")
	parser.add_argument("-m", "--metadata",
		help="Tab-separated metadata for genotypes, if available. "
		+ "First column contains identifiers and should match " 
		+ "genotype IDs from first row of input datafile.")
	parser.add_argument("-s", "--shift", default=1, type=int,
		help="Number of columns between first column and columns with genotypes.")
	parser.add_argument("-a", "--algo", choices=['umap', 'tsne'], default='umap',
		help="Algorithm for dimensionality reduction: umap (default) or tsne.")
	parser.add_argument("-p", "--perplexity", default=20, type=int,
		help="tSNE perplexity parameter.")

	args = parser.parse_args()

	if os.path.exists(args.input):
		args.input = os.path.realpath(args.input)
	else:
		raise FileNotFoundError(args.input)

	if args.metadata:
		if os.path.exists(args.metadata):
			args.metadata = os.path.realpath(args.metadata)
		else:
			raise FileNotFoundError(args.metadata)

	return args


def makeIntSeries(series):
	'''Convert series of genotype to series of integers'''
	try:
		nuc = list(set([
			y for x in set(series.dropna().values) for y in (x[0], x[-1])
		]).intersection({'A', 'C', 'G', 'T'}))
	except:
		raise Exception(set(series.values))
	d = dict(zip(
		[f"{x}{y}" for x in nuc for y in nuc],
		[-1, 0, 0, 1]
	))
	return [d.get(x, pandas.NA) for x in series.values]


def calcDist(df):
	'''Return IBS distances between all samples as a dataframe'''
	l = []
	for x in df.index:
		a = df.loc[x]
		for y in df.index:
			if x > y:
				b = df.loc[y]
				nna = pandas.notna(a) & pandas.notna(b)
				l.append([x, y, sum(abs(a[nna].values - b[nna].values)) / (2 * len(a[nna])) ])
	return pandas.DataFrame.from_records(l, columns=['First','Second','IBSdist']).sort_values('IBSdist')


def main(args):
	'''Read genotypes from tab-separated table to pandas dataframe'''
	df = pandas.read_table(args.input, index_col=0).T
	dfInt = df.iloc[args.shift:, :].apply(makeIntSeries)

	''' Create square matrix of IBS distances '''
	dfDist = calcDist(dfInt)
	dfDist = pandas.pivot(dfDist, index='First', columns='Second', values='IBSdist')
	x = dfDist.transpose().to_numpy()
	x = x[~np.isnan(x)]
	dist_sq = squareform(x, checks=True)
	labels = list(dfDist.columns) + [dfDist.index[-1]]
	dfSQ = pandas.DataFrame(dist_sq, columns=labels, index=labels)

	''' Read metadata '''
	dfMeta = pandas.read_table(args.metadata)
	color_col_name = dfMeta.columns[1]
	color_count = len(set(dfMeta[color_col_name].values))

	use_styles = False
	if len(dfMeta.columns) > 2:
		use_styles = True
		style_col_name = dfMeta.columns[2]

	
	if args.algo == 'umap':
		'''UMAP'''
		reducer = umap.UMAP()
		embedding = reducer.fit_transform(dfSQ)
		dfUMAP = pandas.DataFrame(embedding, columns=['umap1','umap2'])
		dfUMAP['ID'] = labels
		dfUMAP = dfUMAP.merge(dfMeta, how='inner', on='ID')
		dfUMAP.to_csv("grape.umap.tsv")
		color_count = len(set(dfUMAP[color_col_name].values))
		if use_styles:
			relplot(data=dfUMAP, x='umap1', y='umap2', hue=color_col_name,
				palette=color_palette(palette="Set3",n_colors=color_count),
				style=style_col_name)
		else:
			relplot(data=dfUMAP, x='umap1', y='umap2', hue=color_col_name,
				palette=color_palette(palette="Set3",n_colors=color_count))
	else:
		''' Perform tSNE'''
		x_e_pca = TSNE(init='pca', perplexity=args.perplexity, n_jobs=1).fit_transform(dfSQ.iloc[:,1:])
		dfTSNE = pandas.DataFrame(x_e_pca, columns=['tsne1','tsne2'])
		dfTSNE['ID'] = labels
		dfTSNE = dfTSNE.merge(dfMeta, how='inner', on="ID")
		dfTSNE.to_csv("grape.tsne.pca.tsv")
		if use_styles:
			relplot(data=dfTSNE, x='tsne1',y='tsne2', hue=color_col_name,
				palette=color_palette(palette="Set3",n_colors=color_count),
				style=style_col_name)
		else:
			relplot(data=dfTSNE, x='tsne1',y='tsne2', hue=color_col_name,
				palette=color_palette(palette="Set3",n_colors=color_count))

	'''Save images'''
	pyplot.savefig(f"grape.{args.algo}.png", dpi=300)
	pyplot.savefig(f"grape.{args.algo}.svg")


if __name__ == '__main__':
	main(parse_args())
