import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import matplotlib.gridspec as gridspec
import seaborn as sns
from cellCnn.utils import mkdir_p, create_graph
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.stats import gaussian_kde, ks_2samp, ttest_ind
import igraph
#from lifelines.estimation import KaplanMeierFitter

					
def clean_axis(ax):
	ax.get_xaxis().set_ticks([])
	ax.get_yaxis().set_ticks([])
	for sp in ax.spines.values():
		sp.set_visible(False)


def plot_results_2class(results, samples, phenotypes, labels, outdir,
						plot_filter_weights=True, percentage_drop=.2, thres=.3,
						group_a='group a', group_b='group b'):

	# create the output directory
	mkdir_p(outdir)

	# first encode the sample and sample-phenotype for each cell
	sample_sizes = []
	per_cell_ids = []
	for i, x in enumerate(samples):
		sample_sizes.append(x.shape[0])
		per_cell_ids.append(i * np.ones(x.shape[0]))

	# read and plot the selected filters
	nmark = samples[0].shape[1]
	if results['selected_filters'] is None:
		print 'Loading the weights of the best network.'
		filters = results['w_best_net'][:,range(nmark)+[-1]]
	else:
		print 'Loading the weights of consensus filters.'
		filters = results['selected_filters'][:,range(nmark)+[-1]]
	
	if plot_filter_weights:
		plt.figure(figsize=(10, 10))
		ax = sns.heatmap(pd.DataFrame(filters, columns=labels+['output']),
						 robust=True)
		plt.xticks(rotation=90)
		plt.yticks(rotation=0)
		ax.tick_params(axis='both', which='major', labelsize=16)
		plt.tight_layout()
		fig_path = os.path.join(outdir, 'filter_weights.pdf')
		plt.savefig(fig_path)
		plt.close()

	# now select the "good" filters
	dist = results['dist']
	dist = np.max(dist, axis=1)
	sorted_idx = np.argsort(dist)[::-1]
	dist = dist[sorted_idx]
	keep_idx = [sorted_idx[0]]
	for i in range(1, dist.shape[0]):
		if ((dist[i-1] - dist[i]) < percentage_drop * dist[i-1]):
			keep_idx.append(sorted_idx[i])
		else:
			break
 
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

		t = thres * np.max(g)
		condition = g > t
		x1 = x[condition]
		z1 = z[condition]
		g1 = g[condition]
		k = 10

		print 'Creating a k-NN graph with %d/%d cells...' % (x1.shape[0], x.shape[0])
		G = create_graph(x1, k, g1)
		print 'Identifying cell communities...'
		cl = G.community_fastgreedy()
		communities = np.array(cl.as_clustering().membership)
		num_clusters = len(set(communities))
		scores = np.zeros(num_clusters)
		for j in range(num_clusters):
			scores[j] = np.mean(g1[communities == j])

		# keep the "good" communities
		sorted_idx = np.argsort(scores)[::-1]
		scores = scores[sorted_idx]
		keep_idx_comm = [sorted_idx[0]]
		for i in range(1, num_clusters):
			if ((scores[i-1] - scores[i]) < percentage_drop * scores[i-1]):
				keep_idx_comm.append(sorted_idx[i])
			else:
				break

		for com in keep_idx_comm:
			xc = x1[communities == com]
			zc = z1[communities == com]

			ks_list = []
			for j in range(nmark):
				ks = ks_2samp(xc[:,j], x[:,j])
				ks_list.append('KS = %.2f' % ks[0])

			fig_path = os.path.join(outdir, 'selected_population_distribution_filter_%d_cluster_%d.pdf' % (i_filter, com))
			plot_marker_distribution([x, xc], ['all cells', 'selected'],
								 	labels, grid_size=(4,9), ks_list=ks_list, figsize=(24,10),
									colors=['blue', 'red'], fig_path=fig_path, hist=True)

			# additionally, plot a boxplot of per class frequencies
			freq_a, freq_b = [], []
			for ii, (n, yy) in enumerate(zip(sample_sizes, phenotypes)):
				freq = 1. * np.sum(zc == ii) / sample_sizes[ii]
				if yy == 0:
					freq_a.append(freq)
				else:
					freq_b.append(freq)

			# perform a t-test
			_t, pval = ttest_ind(freq_a, freq_b)
		
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
			plt.ylim(0, np.max(freq_a + freq_b) + 0.05)
			sns.despine()
			plt.tight_layout()
			fig_path = os.path.join(outdir, 'selected_population_boxplot_filter_%d_cluster_%d.pdf' % (i_filter, com))
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
				#if seq_index > 0:
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


def plot_single_tsne(z, feat, fig_path, s=2, plot_contours=False):

	sns.set_style('white')
	fig, ax = plt.subplots(figsize=(5,5))
	if plot_contours:
		sns.kdeplot(z[:,0], z[:,1], colors='lightgray', cmap=None, linewidths=0.5)
	im = ax.scatter(z[:, 0], z[:, 1], s=s, marker='o', c=feat, vmin=np.percentile(feat, 1),
					cmap=cm.jet,
					alpha=0.5, edgecolors='face', vmax = np.percentile(feat, 99))
	# magic parameters from http://stackoverflow.com/questions/16702479/matplotlib-colorbar-placement-and-size
	plt.colorbar(im, fraction=0.046, pad=0.04)
	clean_axis(ax)
	ax.grid(False)
	sns.despine()
	#plt.legend(loc="upper left", markerscale=20., scatterpoints=1, fontsize=10)
	#plt.xlabel('tSNE dimension 1', fontsize=20)
	#plt.ylabel('tSNE dimension 2', fontsize=20)
	plt.savefig(fig_path, format='eps')
	plt.clf()
	plt.close()


def tsne_map(z, c, fig_path, colors=None, s=2, suffix='png'):
	c = np.squeeze(c)
	if colors is None:
		colors = sns.color_palette("Set2", len(np.unique(c)))

	sns.set_style('white')
	fig, ax = plt.subplots(figsize=(5,5))
	#sns.kdeplot(z[:,0], z[:,1], colors='lightgray', cmap=None, linewidths=0.5)
	#ax = add_contour(z[c==0], ax)
	for i in np.unique(c):
		#if i > 0:
		plt.scatter(z[c==i, 0], z[c==i, 1], s=s, marker='o', c=colors[i],
						edgecolors='face', label=str(i))

	clean_axis(ax)
	ax.grid(False)

	plt.legend(loc="upper left", markerscale=5., scatterpoints=1, fontsize=10)
	plt.xlabel('tSNE axis 1', fontsize=20)
	plt.ylabel('tSNE axis 2', fontsize=20)
	sns.despine(left=True, bottom=True)
	sns.despine()
	plt.savefig(fig_path + '.%s' % suffix, format=suffix)
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
	plt.xlabel('tSNE dimension 1', fontsize=20)
	plt.ylabel('tSNE dimension 2', fontsize=20)
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


def plot_histograms(datalist, labels, fig_path=None, letter_size=16):
	nmark = len(labels)
	colors = sns.color_palette("Set2", n_colors=len(datalist))
 
	for seq_index in range(nmark):
		fig = plt.figure()
		for ii, x in enumerate(datalist):
			plt.hist(x[:,seq_index], 100, normed=1, facecolor=colors[ii], alpha=0.5)

		plt.title(labels[seq_index])
		if fig_path is not None:
			plt.savefig(fig_path+'_'+labels[seq_index]+'.png')
			plt.clf()  
			plt.close()
		else:
			plt.show()

def plot_nn_weights(w, x_labels, y_labels, fig_path, row_linkage=None, fig_size=(10, 3)):
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


def visualize_results(res, outdir, prefix,
					  plots=['consensus', 'clustering_results'],
					  format='png'):

	w_cons = res['selected_filters']
	w_best_net = res['w_best_net']
	labels = res['labels']
	
	# plot the consensus profiles
	if 'consensus' in plots:
		if len(w_cons.shape) == 1:
			w_cons = w_cons.reshape(1,-1)
			figsize = (10, 2)
		else:
			figsize=(10, 3)
			
		stop = len(labels + ['bias','out'])
		plt.figure(figsize=figsize)
		ax = sns.heatmap(pd.DataFrame(w_cons[:,:stop], columns=labels + ['bias','out']),
						robust=True, yticklabels=False)
		plt.xticks(rotation=90)
		ax.tick_params(axis='both', which='major', labelsize=20)
		plt.tight_layout()
		fig_path = os.path.join(outdir, prefix+'_consensus.'+format)
		plt.savefig(fig_path, format=format)
		plt.close()
 
	# plot the best network weights
	if 'best_net' in plots:
		if len(w_best_net.shape) == 1:
			w_best_net = w_best_net.reshape(1,-1)
			figsize = (10, 2)
		else:
			figsize=(10, w_best_net.shape[0]+1)
			
		stop = len(labels + ['bias','out'])   
		plt.figure(figsize=figsize)
		ax = sns.heatmap(pd.DataFrame(w_best_net[:,:stop], columns=labels + ['bias','out']),
						robust=True)
		plt.xticks(rotation=90)
		ax.tick_params(axis='both', which='major', labelsize=20)
		plt.tight_layout()
		fig_path = os.path.join(outdir, prefix+'_best_net.'+format)
		plt.savefig(fig_path, format=format)
		plt.close()
	
		  
	# plot all the discovered signatures    
	if 'clustering_results' in plots:
		
		cl_res = res['clustering_result']

		if cl_res is not None:
			w_full = cl_res['w']
			Z = cl_res['cluster_linkage']
			clusters = cl_res['cluster_assignments']
	
			
			plt.figure(figsize=(3,2))
			clmap = sns.clustermap(pd.DataFrame(w_full, columns=labels+['bias', 'out']),
								method='average', metric='cosine', row_linkage=Z,
								col_cluster=False, robust=True, yticklabels=clusters)
			plt.setp(clmap.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
			plt.setp(clmap.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
			clmap.cax.set_visible(False)
			fig_path = os.path.join(outdir, prefix+'_all_filters.'+format)
			plt.savefig(fig_path, format=format)
			plt.close()
			
		else:
			sys.stderr.write("Clustering was not performed.\n")
			sys.exit(-1)

'''
def plot_KM(stime, censor, g1, pval, figname, format='eps'):
	sns.set_style('white')
	kmf = KaplanMeierFitter()        
	f, ax = plt.subplots(figsize=(3, 3))
	np.set_printoptions(precision=2, suppress=False)
	kmf.fit(stime[g1], event_observed=censor[g1], label=["high-risk group"])
	kmf.plot(ax=ax, ci_show=False, show_censors=True)
	kmf.fit(stime[~g1], event_observed=censor[~g1], label=["low-risk group"])
	kmf.plot(ax=ax, ci_show=False, show_censors=True)
	ax.grid(b=False)
	sns.despine()
	plt.ylim(0,1)
	plt.xlabel("time", fontsize=14)
	plt.ylabel("survival", fontsize=14)
	plt.text(0.7, 0.85, 'pval = %.2e' % (pval), fontdict={'size': 12},
			horizontalalignment='center', verticalalignment='center',
			transform=ax.transAxes) 
	plt.xticks(rotation=45)
	for item in (ax.get_xticklabels() + ax.get_yticklabels()):
		item.set_fontsize(10)
	plt.tight_layout()
	plt.savefig(figname, format=format)
	plt.close()
'''

def organize_plates(x, y, int_ctype, le, params, yi=1, stim_name='GM-CSF',
					which_filter='max'):

	short_types = {'cd14+hladrmid':'monocytes', 'cd14-hladrmid':'monocytes',
				'cd14+surf-':'surf-', 'cd14-surf-':'surf-',
				'cd4+':'CD4+', 'cd8+':'CD8+', 'dendritic':'dendritic',
				'igm+':'B cells', 'igm-':'B cells', 'nk':'NK cells'}

	if which_filter == 'max':
		w_pos = params[np.argmax(params[:,-1])]
	else:
		w_pos = params[np.argmin(params[:,-1])]
	W = w_pos[:-2]
	b = w_pos[-2]

	df_accum = pd.DataFrame(columns=['cell filter activity', 'condition', ''])
	ctype = np.asarray([short_types[le.inverse_transform(idx)] for idx in int_ctype])
		
	# keep only the class of interest
	x, ctype = x[y == yi], ctype[y == yi]
	act_cnn = np.sum(W.reshape(1,-1) * x, axis=1) + b

	# hack: explicitly cut out the extreme 5% percentiles (outliers) for each cell type
	# because they are not handled properly by the boxplot function
	for curr_ctype in np.unique(short_types.values()):
		curr_act = act_cnn[ctype == curr_ctype]
		if len(curr_act) > 0:
			lq, uq = np.percentile(curr_act, 5), np.percentile(curr_act, 95)
			curr_act[curr_act < lq] = lq
			curr_act[curr_act > uq] = uq
			act_cnn[ctype == curr_ctype] = curr_act
		
	df_stim = pd.DataFrame(columns=['cell filter activity', 'condition', ''])
	df_stim['cell filter activity'] = act_cnn
	if yi == 1:
		df_stim['condition'] = [stim_name] * len(act_cnn)
	else:
		df_stim['condition'] = ['control'] * len(act_cnn)
	df_stim[''] = ctype
	df_accum = df_accum.append(df_stim)
 
	return df_accum


def plot_PBMC_boxplots(x, y, int_ctype, le, params, outdir, stim_name,
						which_filter='max', format='png'):
	
	palette = sns.color_palette("Set1", n_colors=10)
	df_stim = organize_plates(x, y, int_ctype, le, params, yi=1,
								stim_name=stim_name, which_filter=which_filter)
	df_unstim = organize_plates(x, y, int_ctype, le, params, yi=0,
								stim_name=stim_name, which_filter=which_filter)
	df = pd.concat([df_stim, df_unstim])
		
	plt.figure()
	sns.set(style="ticks")

	cols = ['CD4+', 'CD8+', 'NK cells', 'B cells', 'monocytes', 'dendritic', 'surf-']
	df.condition = df.condition.astype("category")
	df.condition.cat.set_categories([stim_name, 'control'], inplace=True)
	ax = sns.boxplot(x="", y="cell filter activity", hue="condition",
					data=df.sort(["condition"]), order = cols,
					palette=palette, whis='range', sym='')
	ax.legend(prop={'size':19})             

	for item in ([ax.xaxis.label, ax.yaxis.label] +
			ax.get_xticklabels() + ax.get_yticklabels()):
		item.set_fontsize(20)
	#plt.tight_layout()
	sns.despine(offset=10, trim=True)
	if which_filter =='max':
		fig_path = os.path.join(outdir, stim_name+'_boxplot_positive.'+format)
	else:
		fig_path = os.path.join(outdir, stim_name+'_boxplot_negative.'+format)
	plt.savefig(fig_path, format=format)
	plt.clf()
	plt.close()
	
	
def plot_benchmark_PR_curves(r_cnn, p_cnn, r_outlier, p_outlier, r_mean, p_mean,
							 r_sc, p_sc, nblast, plotdir, key):
		
	sns.set(style="white")
	curr_palette = sns.color_palette()
	col1 = curr_palette[2]
	col2 = curr_palette[1]
	col3 = curr_palette[0]
	col4 = curr_palette[3]
			
	plt.clf()
	f, ax = plt.subplots()
	plt.plot(r_cnn, p_cnn, c=col1, label='CellCnn')
	plt.plot(r_outlier, p_outlier, c=col2, label='outlier')
	plt.plot(r_mean, p_mean, c=col3, label='mean')
	plt.plot(r_sc, p_sc, c=col4, label='sc')
	plt.xlabel('Recall', fontsize=28)
	plt.ylabel('Precision', fontsize=28)
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.05])
	plt.legend(loc='center left' , prop={'size':24})
	for item in (ax.get_xticklabels() + ax.get_yticklabels()):
		item.set_fontsize(24)
	plt.tight_layout()
	sns.despine()
	mkdir_p(plotdir)
	plt.savefig(os.path.join(plotdir, str(nblast)+'_PR_curve.eps'), format='eps')
	plt.close()


def plot_CellCnn_PR_curves(prec, recall, seq, seq_labels, nclust, plotdir, key):
	sns.set(style="white")
	curr_palette = sns.color_palette("Set1", n_colors=len(seq))

	plt.clf()
	f, ax = plt.subplots()
	for i, nblast in enumerate(seq):
		if (seq_labels == []) or (nclust is None):
			plt.plot(recall[nblast], prec[nblast])
		else:
			if nclust[nblast] == (1,1):
				plt.plot(recall[nblast], prec[nblast], c=curr_palette[i],
						linestyle = '-',
						label=seq_labels[i])
			else:
				plt.plot(recall[nblast], prec[nblast], c=curr_palette[i],
						linestyle = '--',
						label=seq_labels[i] + ' (%d/%d)' % nclust[nblast])

	
	plt.xlabel('Recall', fontsize=28)
	plt.ylabel('Precision', fontsize=28)
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.05])
	plt.legend(loc='center left', prop={'size':20})
	
	for item in (ax.get_xticklabels() + ax.get_yticklabels()):
		item.set_fontsize(24)
	plt.tight_layout()
	sns.despine()
	mkdir_p(plotdir)
	plt.savefig(os.path.join(plotdir, key+'_CellCnn_PRcurve.eps'),
				format='eps')
	plt.close()


def return_FP(y_true, y_pred, TP):
	idx = np.argsort(y_pred)[::-1]
	TP_count = 0
	tot_count = 0
	while TP_count < TP:
		tot_count += 1
		TP_count += y_true[idx[tot_count-1]]
	return (tot_count - TP_count)


def return_FP_cutoff(y_true, y_pred, cutoff):
	pos_true = y_true[y_pred > cutoff]
	P_count = length(pos_true)
	FP_count = np.sum(pos_true == 0)
	return FP_count, P_count

	
def plot_barcharts(y_true, cnn_pred , outlier_pred, mean_pred, sc_pred,
					nblast, label, plotdir, key, at_recall=0.8,
					include_citrus=False, cutoff=None):

	if cutoff is None:
		ntop = int(at_recall * nblast)
		count_h_cnn = return_FP(y_true, cnn_pred, ntop)
		count_h_outlier = return_FP(y_true, outlier_pred, ntop)
		count_h_mean = return_FP(y_true, mean_pred, ntop)
		count_h_sc = return_FP(y_true, sc_pred, ntop)
		count_tot_cnn = ntop + count_h_cnn
		count_tot_outlier = ntop + count_h_outlier
		count_tot_mean = ntop + count_h_mean
		count_tot_sc = ntop + count_h_sc
	else:
		count_h_cnn, count_tot_cnn = return_FP_cutoff(y_true, cnn_pred, cutoff)
		count_h_outlier, count_tot_outlier = return_FP_cutoff(y_true, outlier_pred, cutoff)
		count_h_mean, count_tot_mean = return_FP_cutoff(y_true, mean_pred, cutoff)
		count_h_sc, count_tot_sc = return_FP_cutoff(y_true, sc_pred, cutoff)
	
	methods = ['CellCnn', 'outlier', 'mean', 'sc']
	if include_citrus:
		methods.append('Citrus')
		
	n_method = len(methods)
	arr = np.zeros((n_method, 2), dtype=float)
	arr[0] = np.array([count_h_cnn, count_tot_cnn])
	arr[1] = np.array([count_h_outlier, count_tot_outlier])
	arr[2] = np.array([count_h_mean, count_tot_mean])
	arr[3] = np.array([count_h_sc, count_tot_sc])
	
	# only include if we already have results from a Citrus run in R
	if include_citrus:
		if (label == 'AML') and (nblast > 3000):
			arr[4] = np.array([0, 1])
		else:
			arr[4] = np.array([1, 1])

	for i in range(n_method):
		arr[i] /= arr[i,-1]
	
	df_count = pd.DataFrame(arr, columns = ['ctrl', 'total'])
	df_count['names'] = methods
			
	# now plot the results from the dataframe
	sns.set(style="white")
	f, ax = plt.subplots(figsize=(6, 8))
	#cp = sns.color_palette("Set1", n_colors=8, desat=.5)
	curr_palette = sns.color_palette()
	col1 = curr_palette[2]
	col2 = curr_palette[0]
	
	sns.barplot(x="names", y='total', data=df_count, label=label, color=col1)
	bottom_plot = sns.barplot(x="names", y='ctrl', data=df_count,
							 label='healthy', color=col2)
		
	topbar = plt.Rectangle((0,0),1,1, fc=col1, edgecolor = 'none')
	bbar = plt.Rectangle((0,0),1,1, fc=col2,  edgecolor = 'none')
	l = plt.legend([bbar, topbar], ['healthy', label],
					loc=1, ncol = 2, prop={'size':18})
	l.draw_frame(False)
		
	ax.set(ylim=(0, 1.1), xlabel="",
			ylabel="frequency of selected cell states")
	plt.yticks([.25, 0.5, .75, 1])
	plt.xticks(rotation=45)
	sns.despine(left=True, bottom=True)
		
	#Set fonts to consistent size
	for item in ([bottom_plot.xaxis.label, bottom_plot.yaxis.label] +
				 bottom_plot.get_xticklabels() + bottom_plot.get_yticklabels()):
		item.set_fontsize(22)
	plt.tight_layout()
	mkdir_p(plotdir)
	plt.savefig(os.path.join(plotdir, str(nblast)+'_barcharts_top_cells.eps'), format='eps')
	plt.close()  
 
				   
def scatterplot(x, y, z, xtick_labels, ytick_labels, xlabel, ylabel, plotdir,
				prefix): 
	sns.set(style="white")    
	plt.figure()
	ax = plt.subplot()
	ax.scatter(x, y, s=100, marker='o', c=z, cmap=cm.jet,
			   edgecolors='face', vmin=0, vmax=1)
	plt.grid()
	plt.xticks(range(len(xtick_labels)), xtick_labels)
	plt.yticks(range(len(ytick_labels)), ytick_labels)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	sns.despine(left=True, bottom=True)
	for item in ([ax.xaxis.label, ax.yaxis.label] +
				 ax.get_xticklabels() + ax.get_yticklabels()):
		item.set_fontsize(16)
	plt.tight_layout()

	plt.savefig(os.path.join(plotdir, prefix+'_scatter.eps'), format='eps')
	plt.clf()
	plt.close()

from sklearn.neighbors import NearestNeighbors
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


def produce_tsne_plot(lookup, filter_weights, outdir, labels,
						yi=1, key='Dasatinib', format='png'):

	W = filter_weights[:-2]
	b = filter_weights[-2]

	val = lookup[key]
	x, y, int_ctype = val

	# z-transform the intracellular protein expressions that will be processed by the network
	x_cnn = x[:,10:].copy()
	x_cnn = StandardScaler().fit_transform(x_cnn)
	
	# keep only the stimulated class
	x = x[y == yi]
	x_cnn = x_cnn[y == yi]
	act_cnn = rectify(np.sum(W.reshape(1, -1) * x_cnn, axis=1) + b)
	 
	# create a tsne plot
	model = TSNE(n_components=2, random_state=0)
	z = model.fit_transform(StandardScaler().fit_transform(x[:,:10]))
	np.save(os.path.join(outdir, key+'_tsne_projection.npy'), z)
	#z = np.load(os.path.join(outdir, key+'_tsne_projection.npy'))
	n_clusters=20
	cl_pred = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(z)

	# plot the tSNE map
	plt.figure(figsize=(3, 3))
	ax = plt.subplot(aspect='equal')
	ax.scatter(z[:,0], z[:,1], s=2, marker='o', c=act_cnn, cmap=cm.jet,
				   alpha=0.5, edgecolors='face', vmin=0, 
					vmax=np.percentile(act_cnn[act_cnn>0], 95))
	#print np.percentile(act_cnn[act_cnn>0], 95)
	plt.xlim(-15, 15)
	plt.ylim(-15, 15)
	ax.axis('off')
	ax.axis('tight')
	plt.savefig(os.path.join(outdir, key + 'tsne.'+ format), format=format)
	plt.clf()
	plt.close()
	
	# and the clusters
	plt.figure(figsize=(3, 3))
	ax = plt.subplot(aspect='equal')
	ax.scatter(z[:,0], z[:,1], s=2, marker='o', c=cl_pred, cmap=cm.jet,
					alpha=0.3, edgecolors='face')
	txts = []
	for i in range(n_clusters):
		# position of each label
		xtext, ytext = np.median(z[cl_pred == i, :], axis=0)
		txt = ax.text(xtext, ytext, str(i), fontsize=10)
		txts.append(txt)
	
	plt.ylim(-15, 15)
	ax.axis('off')
	ax.axis('tight')
	plt.savefig(os.path.join(PLOT_PATH, key + 'tsne_clusters.' + format), format=format)
	plt.clf()
	plt.close()
	
	''' set the corresponding clusters per cell type
	y_monocytes = np.logical_or(cl_pred == 5, cl_pred == 8, cl_pred == 16)
	y_dendritic = cl_pred == 10

	# selected biaxial plots #
	##########################

	# indices for CD45, CD123
	idx = [0, 9]
	x_dendritic = x[:,idx]
	labels_dendritic = [labels[i] for i in idx]
	plot_biaxial(x_dendritic, y_dendritic, labels_dendritic, outdir, 'CD123')
	
	# indices for CD45, CD33
	idx = [0, 6]
	x_monocytes = x[:,idx]
	labels_monocytes = [labels[i] for i in idx]
	plot_biaxial(x_monocytes, y_monocytes, labels_monocytes, outdir, 'CD33')
	'''


def plot_biaxial(x, y, labels, outdir, prefix, palette=None):
	sns.set_style('white')
	if palette is None:
		palette = np.array(sns.color_palette("Set2", 2))
	plt.figure(figsize=(3, 3))
	ax = plt.subplot(aspect='equal')
	ax.scatter(x[y==0,0], x[y==0,1], s=2, alpha=0.5,
			   c=palette[0], edgecolors='face')
	ax.scatter(x[y==1,0], x[y==1,1], s=2, alpha=0.5,
			   c=palette[1], edgecolors='face')

	plt.ylim(-1, 5)
	plt.xlabel(labels[0], fontsize=18)
	plt.ylabel(labels[1], fontsize=18)
	plt.tight_layout()
	sns.despine()
	plt.savefig(os.path.join(outdir, prefix + '_biaxial.eps'),
				format='eps')
	plt.clf()
	plt.close()


