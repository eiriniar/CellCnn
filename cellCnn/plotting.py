import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import ImageGrid
import seaborn as sns
from cellCnn.utils import mkdir_p
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from collections import Counter

try:
	from cellCnn.utils import create_graph
except ImportError:
	pass


def clean_axis(ax):
	ax.get_xaxis().set_ticks([])
	ax.get_yaxis().set_ticks([])
	for sp in ax.spines.values():
		sp.set_visible(False)

def plot_results_2class(results, samples, phenotypes, labels, outdir,
						percentage_drop_filter=.2, filter_response_thres=.5,
						group_a='group a', group_b='group b',
						clustering=None, add_filter_response=False,
						percentage_drop_cluster=.1, stat_test='ttest',
						min_cluster_freq=0.2, plot_tsne=True, tsne_ncell=3000):

	''' stat_test: 'ttest' | 'mannwhitneyu'
		clustering: None | 'dbscan' | 'louvain'
	'''

	# create the output directory
	mkdir_p(outdir)

	# numbe rof measured markers
	nmark = samples[0].shape[1]

	# plot the filter weights of the best network
	w_best = results['w_best_net'][:,range(nmark)+[-1]]
	fig_path = os.path.join(outdir, 'best_net_weights.pdf')
	plot_nn_weights(w_best, labels+['output'], fig_path, fig_size=(10, 10))

	# plot the selected filters
	if results['selected_filters'] is not None:
		
		print 'Loading the weights of consensus filters.'
		w = results['selected_filters'][:,range(nmark)+[-1]]
		fig_path = os.path.join(outdir, 'consensus_filter_weights.pdf')
		plot_nn_weights(w, labels+['output'], fig_path, fig_size=(10, 10))
		filters = w
	else:
		filters = w_best

	# plot the filter clustering
	cl = results['clustering_result']
	cl_w = cl['w'][:,range(nmark)+[-1]]
	fig_path = os.path.join(outdir, 'clustered_filter_weights.pdf')
	plot_nn_weights(cl_w, labels+['output'], fig_path, row_linkage=cl['cluster_linkage'],
					y_labels=cl['cluster_assignments'], fig_size=(10, 10))

	
	# select the discriminative filters
	# based on the validation set 
	if 'dist' in results:
		dist = results['dist']
		dist = np.max(dist, axis=1)

	# if no validation set was provided,
	# select filters based on the magnitude of their output weight
	else:
		dist = abs(filters[:,-1])

	sorted_idx = np.argsort(dist)[::-1]
	dist = dist[sorted_idx]
	keep_idx = [sorted_idx[0]]
	for i in range(1, dist.shape[0]):
		if ((dist[i-1] - dist[i]) < percentage_drop_filter * dist[i-1]):
			keep_idx.append(sorted_idx[i])
		else:
			break
 
	print 'Found %d discriminative filter(s): ' % len(keep_idx), keep_idx

	# encode the sample and sample-phenotype for each cell
	sample_sizes = []
	per_cell_ids = []
	for i, x in enumerate(samples):
		sample_sizes.append(x.shape[0])
		per_cell_ids.append(i * np.ones(x.shape[0]))

	# for each selected filter, plot the selected cell population
	x = np.vstack(samples)
	z = np.hstack(per_cell_ids)

	if results['scaler'] is not None:
		x = results['scaler'].transform(x)

	for i_filter in keep_idx: 
		w = filters[i_filter, :nmark]
		b = filters[i_filter, nmark]
		g = np.sum(w.reshape(1,-1) * x, axis=1) + b
		g = g * (g > 0)

		t = filter_response_thres * np.max(g)
		condition = g > t
		x1 = x[condition]
		z1 = z[condition]
		g1 = g[condition]

		# skip a filter if it does not select any cell
		if x1.shape[0] == 0:
			continue
		else:
			# plot a cell filter response map for the filter
			# do it on a subset of the cells, so that it is relatively fast
			if plot_tsne:
				proj = TSNE(n_components=2, random_state=0)
				x_2D = proj.fit_transform(x[:tsne_ncell])
				fig_path = os.path.join(outdir, 'cell_filter_response_%d.png' % i_filter)
				plot_2D_map(x_2D, MinMaxScaler().fit_transform(g.reshape(-1,1))[:tsne_ncell], fig_path)

		if clustering is None:
			suffix = 'filter_%d' % i_filter
			plot_selected_subset(x1, z1, x, labels, sample_sizes, phenotypes,
				 				outdir, suffix, stat_test, group_a, group_b)
		else:
			if clustering == 'louvain':
				print 'Creating a k-NN graph with %d/%d cells...' % (x1.shape[0], x.shape[0])
				k = 10
				G = create_graph(x1, k, g1, add_filter_response)
				print 'Identifying cell communities...'
				cl = G.community_fastgreedy()
				clusters = np.array(cl.as_clustering().membership)
			else:
				print 'Clustering using the dbscan algorithm...'
				eps = set_dbscan_eps(x1, os.path.join(outdir, 'kNN_distances.png'))
				cl = DBSCAN(eps=eps, min_samples=5, metric='l1')
				clusters = cl.fit_predict(x1)

			# discard outliers, i.e. clusters with very few cells
			c = Counter(clusters)
			cluster_ids = []
			min_cells = int(min_cluster_freq * x1.shape[0])
			for key, val in c.items():
				if val > min_cells:
					cluster_ids.append(key)

			num_clusters = len(cluster_ids)
			scores = np.zeros(num_clusters)
			for j in range(num_clusters):
				cl_id = cluster_ids[j]
				scores[j] = np.mean(g1[clusters == cl_id])

			# keep the communities with high cell filter response
			sorted_idx = np.argsort(scores)[::-1]
			scores = scores[sorted_idx]
			keep_idx_comm = [sorted_idx[0]]
			for i in range(1, num_clusters):
				if ((scores[i-1] - scores[i]) < percentage_drop_cluster * scores[i-1]):
					keep_idx_comm.append(sorted_idx[i])
				else:
					break

			for j in keep_idx_comm:
				cl_id = cluster_ids[j]
				xc = x1[clusters == cl_id]
				zc = z1[clusters == cl_id]
				suffix = 'filter_%d_cluster_%d' % (i_filter, cl_id)
				plot_selected_subset(xc, zc, x, labels, sample_sizes, phenotypes,
				 					outdir, suffix, stat_test, group_a, group_b)


def plot_nn_weights(w, x_labels, fig_path, row_linkage=None, y_labels=None, fig_size=(10, 3)):
	if y_labels is None:
		y_labels = range(w.shape[0])
	
	plt.figure(figsize=fig_size)
	clmap = sns.clustermap(pd.DataFrame(w, columns=x_labels),
							method='average', metric='cosine', row_linkage=row_linkage,
							col_cluster=False, robust=True, yticklabels=y_labels)
	plt.setp(clmap.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
	plt.setp(clmap.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
	clmap.cax.set_visible(False)
	plt.savefig(fig_path)
	plt.clf()
	plt.close()

def plot_selected_subset(xc, zc, x, labels, sample_sizes, phenotypes, outdir, suffix,
						stat_test, group_a, group_b):
				
	ks_list = []
	nmark = x.shape[1]
	for j in range(nmark):
		ks = stats.ks_2samp(xc[:,j], x[:,j])
		ks_list.append('KS = %.2f' % ks[0])

	fig_path = os.path.join(outdir, 'selected_population_distribution_%s.pdf' % suffix)
	plot_marker_distribution([x, xc], ['all cells', 'selected'],
							labels, grid_size=(4,9), ks_list=ks_list, figsize=(24,10),
							colors=['blue', 'red'], fig_path=fig_path, hist=True)

	# additionally, plot a boxplot of per class frequencies
	freq_a, freq_b = [], []
	for i, (n, y_i) in enumerate(zip(sample_sizes, phenotypes)):
		freq = 1. * np.sum(zc == i) / sample_sizes[i]
		if y_i == 0:
			freq_a.append(freq)
		else:
			freq_b.append(freq)

	# perform a statistical test
	if stat_test == 'mannwhitneyu':
		_t, pval = stats.mannwhitneyu(freq_a, freq_b)
	else:
		_t, pval = stats.ttest_ind(freq_a, freq_b)
			
	# make a boxplot with error bars
	box_grade = [group_a] * len(freq_a) + [group_b] * len(freq_b)
	box_data = np.hstack([freq_a, freq_b])
	box = pd.DataFrame(np.array(zip(box_grade, box_data)), columns=['group', 'selected population frequency'])
	box['selected population frequency'] = box['selected population frequency'].astype('float64')

	fig, ax = plt.subplots(figsize=(2.5, 2.5))
	ax = sns.boxplot(x="group", y="selected population frequency", data=box, width=.5, palette=sns.color_palette('Set2'))
	ax = sns.swarmplot(x="group", y="selected population frequency", data=box, color=".25")
	ax.text(.45, .95, 'pval = %.3f' % pval, horizontalalignment='center',
			transform=ax.transAxes, size=12, weight='bold')
	plt.ylim(0, np.max(freq_a + freq_b) + 0.01)
	sns.despine()
	plt.tight_layout()
	fig_path = os.path.join(outdir, 'selected_population_boxplot_%s.pdf' % suffix)
	plt.savefig(fig_path)
	plt.clf()
	plt.close()


def plot_marker_distribution(datalist, namelist, labels, grid_size,
							 fig_path=None, letter_size=16, figsize=(9,9), ks_list=None,
							 colors=None, hist=False):
	nmark = len(labels)
	assert len(datalist) == len(namelist)
	g_i, g_j = grid_size
	
	sns.set_style('white')
	if colors is None:
		colors = sns.color_palette("Set1", n_colors=len(datalist), desat=.5)
	
	fig = plt.figure(figsize=figsize)
	grid = gridspec.GridSpec(g_i, g_j, wspace=0.1, hspace=.6)
	for i in range(g_i):
		for j in range(g_j):
			seq_index = g_j * i + j
			if seq_index < nmark:
				ax = fig.add_subplot(grid[i,j])
				if ks_list is not None:
					ax.text(.5, 1.2, labels[seq_index], fontsize=letter_size, ha='center', transform=ax.transAxes)
					ax.text(.5, 1.02, ks_list[seq_index], fontsize=letter_size-4, ha='center', transform=ax.transAxes)
				else:
					ax.text(.5, 1.1, labels[seq_index], fontsize=letter_size, ha='center', transform=ax.transAxes)
 
				for i_name, (name, x) in enumerate(zip(namelist, datalist)):
					lower = np.percentile(x[:,seq_index], 0.5)
					upper = np.percentile(x[:,seq_index], 99.5)
					if seq_index == nmark - 1:
						if hist:
							plt.hist(x[:,seq_index], np.linspace(lower, upper, 10), color=colors[i_name], label=name, alpha=.5, normed=True)
						else:
							sns.kdeplot(x[:,seq_index], color=colors[i_name], label=name, clip=(lower, upper))
					else:
						if hist:
							plt.hist(x[:,seq_index], np.linspace(lower, upper, 10), color=colors[i_name], label=name, alpha=.5, normed=True)
						else:
							sns.kdeplot(x[:,seq_index], color=colors[i_name], clip=(lower, upper))
				ax.get_yaxis().set_ticks([])
				#ax.get_xaxis().set_ticks([-2, 0, 2, 4])

	#plt.legend(loc="upper right", prop={'size':letter_size})
	plt.legend(bbox_to_anchor=(1.5, 0.9))
	sns.despine()
	if fig_path is not None:
		plt.savefig(fig_path)
		plt.close()
	else:
		plt.show()

def set_dbscan_eps(x, fig_path=None):
	nbrs = NearestNeighbors(n_neighbors=2, metric='l1').fit(x)
	distances, indices = nbrs.kneighbors(x)
	
	if fig_path is not None:
		plt.figure()
		plt.hist(distances[:,1], bins=20)
		plt.savefig(fig_path)
		plt.clf()
		plt.close()

	return np.percentile(distances, 90)

def make_biaxial(train_feat, valid_feat, test_feat, train_y, valid_y, test_y, figpath,
				xlabel=None, ylabel=None, add_legend=False):
	# make the biaxial figure
	sns.set_style('white')
	palette = np.array(sns.color_palette("Set2", 3))
	plt.figure(figsize=(3, 3))
	ax = plt.subplot(aspect='equal')

	# the training samples
	ax.scatter(train_feat[:,0], train_feat[:,1], s=30, alpha=.5,
			   c=palette[train_y],  marker='>', edgecolors='face')

	# the validation samples
	ax.scatter(valid_feat[:,0], valid_feat[:,1], s=30, alpha=.5,
			   c=palette[valid_y], marker=(5, 1), edgecolors='face')

	# the test samples
	ax.scatter(test_feat[:,0], test_feat[:,1], s=30, alpha=.5,
			   c=palette[test_y], marker='o', edgecolors='face')

	# http://stackoverflow.com/questions/13303928/how-to-make-custom-legend-in-matplotlib
	a1 = plt.Line2D((0,1),(0,0), color=palette[0])
	a2 = plt.Line2D((0,1),(0,0), color=palette[1])
	a3 = plt.Line2D((0,1),(0,0), color=palette[2])

	a4 = plt.Line2D((0,1),(0,0), color='k', marker='>', linestyle='', markersize=8)
	a5 = plt.Line2D((0,1),(0,0), color='k', marker=(5,1), linestyle='', markersize=8)
	a6 = plt.Line2D((0,1),(0,0), color='k', marker='o', linestyle='', markersize=8)

	#Create legend from custom artist/label lists
	if add_legend:
		first_legend = plt.legend([a1, a2, a3], ['healthy', 'CN', 'CBF'], fontsize=16, loc=1, fancybox=True)
		plt.gca().add_artist(first_legend)
		plt.legend([a4, a5, a6], ['train', 'valid', 'test'], fontsize=16, loc=4, fancybox=True)

	#plt.xlim(-2, 2)
	#plt.ylim(-2, 2)
	ax.set_aspect('equal', 'datalim')
	ax.margins(0.1)
	
	if xlabel is not None:
		plt.xlabel(xlabel, fontsize=12)
	if ylabel is not None:
		plt.ylabel(ylabel, fontsize=12)

	plt.tight_layout()
	sns.despine()
	plt.savefig(figpath, format='eps')
	plt.clf()
	plt.close()


def plot_tsne_grid(z, x, grid_size, fig_path, labels=None, fig_size=(9,9),
				   suffix='png', point_size=.1):
   
	ncol = x.shape[1]
	if labels is None:
		labels = [str(a) for a in range(ncol)]

	sns.set_style('white')
	fig = plt.figure(figsize=fig_size)
	fig.clf()
	g_i, g_j = grid_size
	grid = ImageGrid(fig, 111, 
					 nrows_ncols=(g_i, g_j),
					 ngrids = ncol,
					 aspect=True,
					 direction="row",
					 axes_pad=(0.15, 0.5),
					 add_all=True,
					 label_mode="1",
					 share_all=True,
					 cbar_location="top",
					 cbar_mode="each",
					 cbar_size="8%",
					 cbar_pad="5%",
					 )
	for seq_index, ax in enumerate(grid):
		ax.text(0, .92, labels[seq_index],
				horizontalalignment='center',
				transform=ax.transAxes, size=20, weight='bold')
		vmin = np.percentile(x[:,seq_index], 1)
		vmax = np.percentile(x[:,seq_index], 99)
		#sns.kdeplot(z[:,0], z[:,1], colors='gray', cmap=None, linewidths=0.5)
		im = ax.scatter(z[:,0], z[:,1], s=point_size, marker='o', c=x[:,seq_index], cmap=cm.jet,
						alpha=0.5, edgecolors='face', vmin=vmin, vmax=vmax)                
		ax.cax.colorbar(im)            
		clean_axis(ax)
		ax.grid(False)
				 
	plt.savefig('.'.join([fig_path, suffix]), format=suffix)
	plt.clf()
	plt.close()


def plot_tsne_selection_grid(z_pos, x_pos, z_neg, vmin, vmax, grid_size, fig_path,
							 labels=None, fig_size=(9,9), suffix='png', text_annot=None):
	ncol = x_pos.shape[1]
	if labels is None:
		labels = [str(a) for a in np.range(ncol)]
	
	fig = plt.figure(figsize=fig_size)
	fig.clf()
	g_i, g_j = grid_size
	grid = ImageGrid(fig, 111, 
					 nrows_ncols=(g_i, g_j),
					 ngrids = ncol,
					 aspect=True,
					 direction="row",
					 axes_pad=(0.15, 0.5),
					 add_all=True,
					 label_mode="1",
					 share_all=True,
					 cbar_location="top",
					 cbar_mode="each",
					 cbar_size="8%",
					 cbar_pad="5%",
					 )
	for seq_index, ax in enumerate(grid):
		ax.text(0, .92, labels[seq_index],
				horizontalalignment='center',
				transform=ax.transAxes, size=20, weight='bold')
		a = x_pos[:, seq_index]
		ax.scatter(z_neg[:,0], z_neg[:,1], s=.5, marker='o', c='lightgray',
					alpha=0.5, edgecolors='face')                
		im = ax.scatter(z_pos[:,0], z_pos[:,1], s=.5, marker='o', c=a, cmap=cm.jet,
						edgecolors='face', vmin=vmin[seq_index], vmax=vmax[seq_index])
		ax.cax.colorbar(im)            
		clean_axis(ax)
		ax.grid(False)
  
	plt.savefig('.'.join([fig_path, suffix]), format=suffix)
	plt.clf()
	plt.close()


def plot_2D_map(z, feat, fig_path, s=2, plot_contours=False):

	sns.set_style('white')
	fig, ax = plt.subplots(figsize=(5,5))
	if plot_contours:
		sns.kdeplot(z[:,0], z[:,1], colors='lightgray', cmap=None, linewidths=0.5)

	if issubclass(feat.dtype.type, np.integer):
		c = np.squeeze(feat)
		colors = sns.color_palette("Set2", len(np.unique(c)))
		for i in np.unique(c):
			plt.scatter(z[c==i, 0], z[c==i, 1], s=s, marker='o', c=colors[i],
						edgecolors='face', label=str(i))
	else:
		im = ax.scatter(z[:, 0], z[:, 1], s=s, marker='o', c=feat, vmin=np.percentile(feat, 1),
						cmap=cm.jet, alpha=0.5, edgecolors='face', vmax=np.percentile(feat, 99))
	
		# magic parameters from http://stackoverflow.com/questions/16702479/matplotlib-colorbar-placement-and-size
		plt.colorbar(im, fraction=0.046, pad=0.04)
	clean_axis(ax)
	ax.grid(False)
	sns.despine()
	if issubclass(feat.dtype.type, np.integer):
		plt.legend(loc="upper left", markerscale=5., scatterpoints=1, fontsize=10)
	plt.xlabel('tSNE dimension 1', fontsize=20)
	plt.ylabel('tSNE dimension 2', fontsize=20)
	plt.savefig(fig_path)
	plt.clf()
	plt.close()


def plot_tsne_per_sample(z_list, data_list, data_labels, fig_dir, fig_size=(9,9),
						density=True, scatter=True, colors=None, pref=''):

	if colors is None:
		colors = sns.color_palette("husl", len(data_list))
	start = 0

	fig, ax = plt.subplots(figsize=fig_size)
	for i, x in enumerate(data_list):
		z = z_list[i]
		ax.scatter(z[:, 0], z[:, 1], s=1, marker='o', c=colors[i],
					alpha=0.5, edgecolors='face', label=data_labels[i])
	clean_axis(ax)
	ax.grid(False)

	plt.legend(loc="upper left", markerscale=20., scatterpoints=1, fontsize=10)
	plt.xlabel('t-SNE dimension 1', fontsize=20)
	plt.ylabel('t-SNE dimension 2', fontsize=20)
	plt.savefig(os.path.join(fig_dir, pref+'_tsne_all_samples.png'), format='png')
	plt.clf()
	plt.close()

	# density plots
	if density:
		for i, x in enumerate(data_list):
			fig = plt.figure(figsize=fig_size)
			z = z_list[i]
			sns.kdeplot(z[:, 0], z[:, 1], n_levels=30, shade=True)
			plt.title(data_labels[i])
			plt.savefig(os.path.join(fig_dir, pref+'tsne_density_%d.png' % i), format='png')
			plt.clf()
			plt.close()

	if scatter:
		for i, x in enumerate(data_list):
			fig = plt.figure(figsize=fig_size)
			z = z_list[i]
			plt.scatter(z[:, 0], z[:, 1], s=1, marker='o', c=colors[i],
						alpha=0.5, edgecolors='face')
			plt.title(data_labels[i])
			plt.savefig(os.path.join(fig_dir, pref+'tsne_scatterplot_%d.png' % i), format='png')
			plt.clf()
			plt.close()
