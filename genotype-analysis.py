#!/usr/bin/env python3
import argparse
import logging
import os
import pandas
import seaborn
import sys
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import TSNE


def makeIntSeries(series):
	logger = logging.getLogger("main")
	try:
		genotypes = set(series.dropna().values)
		nuc = sorted(list(set([
			y for x in genotypes for y in (x[0], x[-1])
		]).intersection({'A', 'C', 'G', 'T'})))
	except:
		logger.error("Error with values: %s", set(series.values))
		raise Exception(set(series.values))
	if len(nuc) > 2:
		logger.warning('More than two alleles: %s', ' '.join(nuc))
	d = dict(zip(
		[f"{x}{y}" for x in nuc for y in nuc],
		[-1, 0, 0, 1]
	))
	return [d.get(x, pandas.NA) for x in series.values]


def makeIntDF(df, args):
	logger = logging.getLogger("main")
	logger.info('Converting genotypes to integer data.')
	if args.transpose:
		dfInt = df.iloc[args.shift:, :].apply(makeIntSeries)
	else:
		dfInt = df.iloc[:, args.shift:].apply(makeIntSeries)
	dfInt.index = list(map(lambda x: x.replace('.', '_'), dfInt.index))
	#dfInt.index = list(map(lambda x: x.replace(' ', '_'), dfInt.index))
	return dfInt


def predGT(a, b):
	if(
		a is pandas.NA
		or b is pandas.NA
		or a is None
		or b is None
		or a == 0
		or b == 0
	):
		return None
	return (a + b) // 2


def predictOffs(dfParInt):
	d = dict()
	for P1 in dfParInt.index:
		for P2 in dfParInt.index:
			if P1 >= P2:
				continue
			predOfGT = [predGT(x, y) for x, y in zip(
				dfParInt.loc[P1].values, dfParInt.loc[P2].values
			)]
			d[f"{P1}.{P2}"] = pandas.array(predOfGT)

	return pandas.DataFrame.from_dict(d, orient='index')


def DixonQ(A):
	A = sorted(A)
	return (A[1] - A[0]) / (A[-1] - A[0])


def getTriosAboveGap(dfTrio):
	logger = logging.getLogger("main")
	DefaultGDistThr = 0.2
	dfTrioGood = dfTrio[dfTrio['Gdist'] < DefaultGDistThr].copy()
	# ['Offspring', 'ppID', 'Parent1', 'Parent2', 'Gdist', 'snpN', 'snpDiff']

	dfTrioGood['Gdiff'] = [0] + list(dfTrioGood['Gdist'][1:].values - dfTrioGood['Gdist'][:-1].values)
	maxDiffIdx = dfTrioGood['Gdiff'].idxmax()
	thrDist = dfTrioGood['Gdist'][maxDiffIdx]
	logger.info("Genotype distance gap: %s", dfTrioGood['Gdiff'][maxDiffIdx])
	logger.info("Genotype distance at threshold: %s", thrDist)
	dfTrioGood = dfTrioGood.query('Gdist < @thrDist').sort_values('Gdist')
	dfTrioGood = dfTrioGood.drop_duplicates(subset=['Offspring'], keep='first')
	logger.info("Parentage trios found: %s", len(dfTrioGood))

	return dfTrioGood


def Qtest(dfTrio):
	q95 = [
		0.97, 0.829, 0.71, 0.625, 0.568, 0.526, 0.493, 0.466, 0.444,
		0.426, 0.41, 0.396, 0.384, 0.374, 0.365, 0.356, 0.349, 0.342,
		0.337, 0.331, 0.326, 0.321, 0.317, 0.312, 0.308, 0.305, 0.301, 0.29
	]
	Q95 = {n:q for n, q in zip(range(3, len(q95) + 3), q95)}
	dfTrio = dfTrio[dfTrio['Gdist'] < 0.2].copy()

	dfTrio['Gdiff'] = [0] + list(dfTrio['Gdist'][1:].values - dfTrio['Gdist'][:-1].values)
	dfTrio['Sign'] = False

	maxDiffIdx = dfTrio['Gdiff'].idxmax()
	thrDist = dfTrio['Gdist'][maxDiffIdx]
	print("Thr Gdiff:", dfTrio['Gdiff'][maxDiffIdx], "Gdist at thr.:", thrDist)

	dfBT = dfTrio[dfTrio['Gdist'] > thrDist]
	size = 1 + min(29, len(dfBT))
	if size < 3:
		print("Too small size of the sample.")
		return dfTrio
	if (len(dfBT) > 29):
		Qs = []
		for _ in range(100):
			thrTest = dfBT.sample(29)['Gdist']
			Qs.append((thrDist - min(thrTest)) / (thrDist - max(thrTest)))
		Q = sum(Qs) / len(Qs)
	else:
		thrTest = dfBT['Gdist']
		Q = (min(thrTest) - thrDist) / (max(thrTest) - thrDist)

	Qt = Q95[size]
	print(f"Sample size: {size}. Qt {Qt}, Q-value {round(Q, 3)}.")
	if Q < Qt:
		print("Insignificant threshold for the gap:", round(dfTrio['Gdiff'][maxDiffIdx], 3))
		return dfTrio
	else:
		dfAT = dfTrio[dfTrio['Gdist'] <= thrDist]
		for i in dfAT.index:
			Q = DixonQ([dfAT.loc[i]['Gdist']] + list(dfBT['Gdist'].values))
			if Q95[size] < Q:
				dfTrio.at[i, 'Sign'] = True

	return dfTrio


def countOpposite(a, b):
	'''Return count of opposing homozygotes and homozygotes'''
	idx_not_na = pandas.notna(a) & pandas.notna(b)
	mult = a[idx_not_na] * b[idx_not_na]
	mult_vc = mult.value_counts()
	ophz = mult_vc.get(-1, 0)  # count 1 * -1
	hz = ophz + mult_vc[1]  # add up counts 1 * 1 and -1 * -1
	return ophz, hz


def opposite(dfInt, dfDist, min_dist=0.05, min_loci=20):
	def check_dist(a, b):
		return dfDist.query('First == @a & Second == @b')['IBSdist'].values[0]

	logger = logging.getLogger('main')
	logger.info('Calculating opposing homozygotes for all possible pairs.')
	opps = []
	for i in dfInt.index:
		for j in dfInt.index:
			if i > j:
				if check_dist(i, j) > min_dist:
					ophz, hz = countOpposite(dfInt.loc[i], dfInt.loc[j])
					if hz > min_loci:
						opps.append([i, j, ophz, hz])
				else:
					logger.debug(f"{i} and {j} are too close, skipping.")
	return pandas.DataFrame.from_records(
		opps, columns=['First', 'Second', 'OH', 'loci']
		).sort_values('OH')


def calcOppNA(dfPred, dfOrf):
	opps = []
	for x in dfPred.index:
		if x.split(".")[1] == x.split(".")[0]:
			continue
		for y in dfOrf.index:
			if y in x.split("."):
				continue
			opps.append([
				x, y, countOpposite(dfPred.loc[x].values, dfOrf.loc[y].values)
				])
	return pandas.DataFrame.from_records(opps, columns=['First', 'Second', 'OH']).sort_values('OH')


def IBSDist(df):
	'''Return IBS distances between all samples as a dataframe'''
	logger = logging.getLogger('main')
	logger.info('Calculating IBS distances.')
	dists = []
	for x in df.index:
		a = df.loc[x]
		for y in df.index:
			if x > y:
				b = df.loc[y]
				nna = pandas.notna(a) & pandas.notna(b)
				dists.append([
					x,
					y,
					sum(abs(a[nna].values - b[nna].values)) / (2 * len(a[nna]))
				])
	return pandas.DataFrame.from_records(dists, columns=['First', 'Second', 'IBSdist']).sort_values('IBSdist')


def NeiDaDist(df):
	'''Return Nei's Da distances between all samples as a dataframe'''
	logger = logging.getLogger('main')
	logger.info('Calculating Nei\'s DA distances.')
	dists = []
	for x in df.index:
		a = df.loc[x]
		for y in df.index:
			if x > y:
				b = df.loc[y]
				nna = pandas.notna(a) & pandas.notna(b)
				dists.append([
					x,
					y,
					1 - sum((a[nna] * b[nna]).map(sqrt) + ( (1 - a[nna]) * (1 - b[nna]) ).map(sqrt)) / len(a[nna])
					])
	return pandas.DataFrame.from_records(dists, columns=['First', 'Second', 'NeiDa']).sort_values('NeiDa')


def triosFromDuos(dfInt, dfDuo):
	'''Return all possible trios and predicted progeny genoypes'''
	logger = logging.getLogger('main')
	logger.info('Predicting parents-offspring trios.')
	trios = []
	d = dict()
	for x in set(dfDuo['First']).union(set(dfDuo['Second'])):
		ops = list(dfDuo[dfDuo['First'] == x]['Second'].values) + list(dfDuo[dfDuo['Second'] == x]['First'].values)
		if len(ops) < 2:
			continue
		testProg = dfInt.loc[x]
		for y in ops:
			for z in ops:
				if y < z or (y == z and not args.no_self):
					predProg = pandas.Series([predGT(a, b) for a, b in zip(
							dfInt.loc[y].values, dfInt.loc[z].values
						)],
						index=testProg.index
					)
					nna = predProg.notna() & testProg.notna()
					diff = sum([0 if(a == b) else 1 for a, b in zip(
						predProg[nna].values, testProg[nna].values
					)])
					snpN = len(predProg[nna])
					if snpN > 0:
						dist = diff / snpN
						trios.append([x, f"{y}.{z}", y, z, dist, snpN, diff])
						d[f"{y}.{z}"] = predProg

	header = ['Offspring', 'ppID', 'Parent1', 'Parent2', 'Gdist', 'snpN', 'snpDiff']
	dfTrios = pandas.DataFrame.from_records(trios, columns=header)
	dfPPGT = pandas.DataFrame.from_dict(d, orient='index')
	return dfTrios.sort_values('Gdist'), dfPPGT


def create_logger():
	logger = logging.getLogger("main")
	logger.setLevel(logging.DEBUG)
	formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)
	ch.setFormatter(formatter)
	logger.addHandler(ch)
	return logger


def dendro(args, df):
	logger = logging.getLogger("main")
	x = df.transpose().to_numpy()
	x = x[~np.isnan(x)]
	# dist_sq = squareform(x, checks=True)
	labels = list(df.columns) + [df.index[-1]]
	logger.info("Plotting dendrograms.")
	dist = ("IBS", "Nei's genetic")[args.nei_distance]
	for method in ("complete", "average", "weighted", "ward"):
		lm = linkage(x, method=method, optimal_ordering=True)
		fig = plt.figure(figsize=(6, args.ratio * 6), dpi=300)
		plt.title(f"{dist} distance dendrogram, {method} clustering")
		dendrogram(lm, orientation='right', labels=labels,
			distance_sort='descending', show_leaf_counts=True)
		plt.savefig(f"{args.prefix}.dendrogram.{method}.png", dpi=300, bbox_inches='tight')
		plt.savefig(f"{args.prefix}.dendrogram.{method}.svg", bbox_inches='tight')
		plt.close()


def pt_to_plink(a):
	if pandas.isna(a):
		return (0, 0)
	if a == -1:
		return (1, 1)
	if a == 0:
		return (1, 2)
	if a == 1:
		return (2, 2)


def create_plink_ped(df):
	logger = logging.getLogger("main")
	logger.info('Converting data to PLINK format.')
	plink_list = []
	for i in df.index:
		out = [z for x in df.loc[i] for z in pt_to_plink(x)]
		plink_list.append([i] + [0] * 4 + out)
	df_ped = pandas.DataFrame.from_records(plink_list)
	df_ped.to_csv(f"{args.prefix}.ped", sep=' ', header=False)


def filter_duo(dfDuo, dfDist):
	idx = []
	for x in dfDuo.index:
		f, s = dfDuo.loc[x][['First', 'Second']].values
		dist = dfDist.query('First == @f & Second == @s')['IBSdist'].values
		if dist < 0.05:
			idx.append(x)
	return dfDuo.drop(idx)


def tnse_viz(dfDistP):
	# Set perplexity parameter here:
	perplexity = args.perplexity

	x = dfDistP.transpose().to_numpy()
	x = x[~np.isnan(x)]
	dist_sq = squareform(x, checks=True)
	labels = list(dfDistP.columns) + [dfDistP.index[-1]]
	dfSQ = pandas.DataFrame(dist_sq, columns=labels, index=labels)

	dfMeta = pandas.read_table(args.metadata)
	if args.subsample:
		dfMeta = dfMeta.iloc[0:args.subsample, :]
	label_col, label_style = dfMeta.columns[1:3]

	x_e_pca = TSNE(init='pca', perplexity=perplexity, n_jobs=1).fit_transform(dfSQ.iloc[:, 1:])
	dfTSNE = pandas.DataFrame(x_e_pca, columns=['tsne1', 'tsne2'])
	dfTSNE['ID'] = labels
	dfTSNE = dfTSNE.merge(dfMeta, how='inner', on="ID")
	dfTSNE.to_csv(f"{args.prefix}.tsne.pca.tsv")

	n_colors = len(set(dfTSNE[label_col].values))
	seaborn.relplot(data=dfTSNE, x='tsne1', y='tsne2', hue=label_col,
		palette=seaborn.color_palette(palette="tab20", n_colors=n_colors),
		style=label_style)
	plt.savefig(f"{args.prefix}.tsne.pca.png", dpi=300)
	plt.savefig(f"{args.prefix}.tsne.pca.svg")
	plt.close()


def main():
	global args
	args = parse_args()
	out_dir = os.path.dirname(args.input)
	logger = create_logger()
	logger.info("Analysis started")

	if args.transpose:
		df = pandas.read_table(args.input, sep=args.separator, index_col=0).T
	else:
		df = pandas.read_table(args.input, sep=args.separator, index_col=0, header=None)

	if args.subsample:
		df = df.iloc[0:args.subsample + args.shift, :]

	if args.gt_separator:
		df = df.replace(to_replace=args.gt_separator, value='', regex=True)

	dfInt = makeIntDF(df, args)
	dfInt.to_csv("%s.int.csv" % args.prefix)

	if args.create_plink:
		create_plink_ped(dfInt)

	# Calculate IBS distance
	dfDist = IBSDist(dfInt)
	dfDist.to_csv("%s.IBSdist.csv" % args.prefix, index=None)
	dfDistP = pandas.pivot(dfDist, index='First', columns='Second', values='IBSdist')
	dfDistP.to_csv("%s.IBSdist.pivot.csv" % args.prefix)

	# Find possible duplicates
	dfDupes = dfDist[dfDist['IBSdist'] < args.dup_threshold]
	if len(dfDupes) > 0:
		dfDupes.to_csv("%s.duplicates.csv" % args.prefix, index=None)
		logger.info("%s duplicates found.", len(dfDupes))
		dfIntNR = dfInt.drop(labels=dfDupes['First'])
	else:
		logger.info("No duplicates found.")
		dfIntNR = dfInt

	# Calculate Nei's DA distance
	if args.nei_distance:
		dfP = 0.5 - dfInt / 2
		dfNei = NeiDaDist(dfP)
		dfNei.to_csv("%s.NeiDa.csv" % args.prefix, index=None)
		dfNeiP = pandas.pivot(dfNei, index='First', columns='Second', values='NeiDa')
		dfNeiP.to_csv("%s.NeiDa.pivot.csv" % args.prefix)
	else:
		dfNeiP = None

	if args.tsne:
		tnse_viz((dfDistP, dfNeiP)[args.nei_distance])

	# Plot dendrograms
	if args.plot_dendrograms:
		dendro(args, (dfDistP, dfNeiP)[args.nei_distance])

	# Calculate IBD / opposing homozygotes
	dfOpp = opposite(dfIntNR, dfDist)
	dfOpp['Err_rate'] = dfOpp['OH']/dfOpp['loci']
	dfOpp = dfOpp.sort_values('Err_rate')
	dfOpp.to_csv("%s.OH.csv" % args.prefix, index=None)
	seaborn.histplot(data=dfOpp, x='OH', binwidth=1)
	plt.savefig("%s.OH.hist.png" % args.prefix, dpi=300)
	plt.close()

	# Calculate mean MAF if not specified by user
	if args.compute_MAF:
		args.MAF = np.mean((1 - abs(dfIntNR.mean())) / 2)
		logger.info(
		"Minor allele frequency (MAF): %s",
		round(args.MAF, 2)
		)

	snpN = len(dfIntNR.columns)
	halfsiblingOH = 0.5 * snpN * args.MAF ** 2 * (1 - args.MAF) ** 2
	logger.info(
		"Predicted opposing homozygotes count for halfsiblings: %s",
		round(halfsiblingOH, 2)
		)

	h = np.histogram(dfOpp[dfOpp['OH'] <= halfsiblingOH * 2]['OH'], bins='auto')
	# 'bins' may be set to int(halfsiblingOH)
	OHthreshold = h[1][np.argmin(h[0])]
	logger.info(
		"Max opposing homozygotes for putative relatives: %s",
		round(OHthreshold, 2)
		)
	dfDuo = dfOpp.query('OH <= @OHthreshold').copy()
	dfDuo = filter_duo(dfDuo, dfDist)
	dfDuo['Orient'] = pandas.NA
	logger.info("Parent-Offspring pairs found: %s", len(dfDuo))

	if len(dfDuo.index) > 1:
		dfTrio, dfPPGT = triosFromDuos(dfIntNR, dfDuo)
		dfTrio = dfTrio.drop_duplicates(subset=['Offspring'], keep='first')
		shortHeader = ['Offspring', 'Parent1', 'Parent2', 'Gdist', 'snpN', 'snpDiff']
		dfTrio[shortHeader].to_csv("%s.trios.all.csv" % args.prefix, index=None)
		dfPPGT.to_csv("%s.PP.GT.csv" % args.prefix)
		dfTrioGood = getTriosAboveGap(dfTrio[shortHeader])
		dfTrioGood.to_csv("%s.trios.good.csv" % args.prefix, index=None)

		'''Try to orient duo when both parents are known.
		May cause erros with siblings, close lineages or clones.
		'''
		for x in dfDuo.index:
			a, b = dfDuo.loc[x].values[:2]
			if a in dfTrioGood['Offspring'].values:
				if b in dfTrioGood.query('Offspring == @a')[['Parent1', 'Parent2']].values:
					# Offspring {a} and parent {b}
					dfDuo.loc[x, 'Orient'] = 'OP'
				else:
					# Parent {a} and offspring {b}
					dfDuo.loc[x, 'Orient'] = 'PO'
			elif b in dfTrioGood['Offspring'].values:
				if a in dfTrioGood.query('Offspring == @b')[['Parent1', 'Parent2']].values:
					# Offspring {b} and parent {a}
					dfDuo.loc[x, 'Orient'] = 'PO'
				else:
					# Parent {b} and offspring {a}
					dfDuo.loc[x, 'Orient'] = 'OP'

	dfDuo.to_csv("%s.duo.csv" % args.prefix, index=None)

	if args.qtest:
		dfTrioQ = Qtest(dfTrio)
		dfTrioQ.to_csv("%s.trios.Qtested.csv" % args.prefix, index=None)

	if args.predict:
		# Predict all possible progeny
		logger.info("Calculating all possible progeny")
		dfAllPP = predictOffs(dfIntNR)
		dfAllPP.to_csv("%s.PPall.GT.csv" % args.prefix)
		logger.info("Possible progeny calculated and written.")

		# Drop known offsprings
		dfIntMissingParentage = dfIntNR.drop(
			labels=dfTrioGood['Offspring'].values, axis='index'
			)

		# Calculate OH between predicted progeny and orphans
		logger.info("Calculating OH between predicted progeny and orphans.")
		dfOpp2 = calcOppNA(dfAllPP, dfIntMissingParentage)
		dfOpp2[dfOpp2['OH'] <= 5].to_csv(
			"%s.OH.withPredicted.csv" % args.prefix, index=None
			)

	logger.info("Analysis finished.")


def parse_args():
	parser = argparse.ArgumentParser(description='Genome statistics')
	parser.add_argument('-i', '--input', required=True,
		help='Input TSV file with genotypes.')
	parser.add_argument('-N', '--subsample', type=int,
		help='Subsample size.')
	parser.add_argument('--no-self', action='store_true',
		help='Do not produce offspring from identical parents.')
	parser.add_argument('--transpose', action='store_true',
		help='To transpose initial data.')
	parser.add_argument('--MAF', type=float, default=0.45,
		help='Average MAF of the SNP set for '
		+ 'opposing homozygotes counts estimations.')
	parser.add_argument('--compute-MAF', action='store_true',
		help='Compute MAF')
	parser.add_argument('--gt-separator',
		help='Symbol separating alleles in genotype, none by default.')
	parser.add_argument('--separator', default='\t',
		help='Symbol separating values in the table. Default: tab')
	parser.add_argument('--shift', type=int, default=1,
		help='Count of columns with meta-data, preceding genotypes.')
	parser.add_argument('--predict', action='store_true',
		help='Predict intermediates.')
	parser.add_argument('--create-plink', action='store_true',
		help='Create ped-file (PLINK pedigry and genotype text file).')
	parser.add_argument('-d', '--plot-dendrograms', action='store_true',
		help='Plot dendrograms.')
	parser.add_argument('-p', '--prefix',
		help='Output prefix.')
	parser.add_argument('--dup-threshold', type=float, default=0.01,
		help='Threshold for duplicate removal, default 0.01.')
	parser.add_argument('--tsne', action='store_true',
		help='Perform tSNE dimensionality scaling.')
	parser.add_argument('-m', '--metadata',
		help='Metadata for tSNE plot formatting.')
	parser.add_argument('--qtest', action='store_true',
		help='Perform Qtest (may be inaccurate).')
	parser.add_argument('--nei-distance', action='store_true',
		help='Use Nei\'s DA distance for tSNE and dendrograms.')
	parser.add_argument('--perplexity', default=20, type=int,
		help='tSNE perplexity, default: 20.')
	parser.add_argument('--ratio', type=float, default=8,
		help='Denrogram ration Height:Width. Default 8.')

	args = parser.parse_args()

	if not os.path.isfile(args.input):
		Exception("File %s not found" % args.input)

	args.input = os.path.abspath(args.input)

	if args.prefix is None:
		args.prefix = os.path.basename(args.input)

	return args


if __name__ == '__main__':
	main()
