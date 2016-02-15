import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import matplotlib.gridspec as gridspec
import seaborn as sns
from cellCnn.utils import mkdir_p
                    
def clean_axis(ax):
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_marker_distribution(datalist, namelist, labels, grid_size,
                             fig_path=None, letter_size=16):
    nmark = len(labels)
    assert len(datalist) == len(namelist)
    g_i, g_j = grid_size
    
    colors = sns.color_palette("Set1", n_colors=len(datalist), desat=.5)
    
    fig = plt.figure()
    grid = gridspec.GridSpec(g_i, g_j, wspace=0.1, hspace=0.05)
    for i in range(g_i):
        for j in range(g_j):
            seq_index = g_j * i + j
            if seq_index < nmark:
                ax = fig.add_subplot(grid[i,j])
                start = .5
                ax.text(start,.85, labels[seq_index],
                    horizontalalignment='center',
                    transform=ax.transAxes, size=letter_size)    
                for i_name, (name, x) in enumerate(zip(namelist, datalist)):
                    lower = np.percentile(x[:,seq_index], 0.5)
                    upper = np.percentile(x[:,seq_index], 99.5)
                    if seq_index == nmark - 1:
                        sns.kdeplot(x[:,seq_index], color=colors[i_name], label=name,
                                    clip=(lower, upper))
                    else:
                        sns.kdeplot(x[:,seq_index], color=colors[i_name],
                                    clip=(lower, upper))
                clean_axis(ax)
    plt.legend(loc="upper right", prop={'size':letter_size})
    if fig_path is not None:
        plt.savefig(fig_path, format='eps')
        plt.close()
    else:
        plt.show()


def visualize_results(res, outdir, prefix,
                      plots=['consensus'],
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
            
        plt.figure(figsize=figsize)
        ax = sns.heatmap(pd.DataFrame(w_cons, columns=labels + ['bias','out']),
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
            
        plt.figure(figsize=figsize)
        ax = sns.heatmap(pd.DataFrame(w_best_net, columns=labels + ['bias','out']),
                        robust=True, yticklabels=False)
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
            clmap.cax.set_visible(False)
            fig_path = os.path.join(outdir, prefix+'_all_filters.'+format)
            plt.savefig(fig_path, format=format)
            plt.close()
            
        else:
            sys.stderr.write("Clustering was not performed.\n")
            sys.exit(-1)



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
    
def plot_barcharts(y_true, cnn_pred , outlier_pred, mean_pred, sc_pred,
                    nblast, label, plotdir, key, at_recall=0.8, include_citrus=False):
    ntop = int(at_recall * nblast)
    count_h_cnn = return_FP(y_true, cnn_pred, ntop)
    count_h_outlier = return_FP(y_true, outlier_pred, ntop)
    count_h_mean = return_FP(y_true, mean_pred, ntop)
    count_h_sc = return_FP(y_true, sc_pred, ntop)
    
    methods = ['CellCnn', 'outlier', 'mean', 'sc']
    if include_citrus:
        methods.append('Citrus')
        
    n_method = len(methods)
    arr = np.zeros((n_method, 2), dtype=float)
    arr[0] = np.array([count_h_cnn, ntop+count_h_cnn])
    arr[1] = np.array([count_h_outlier, ntop+count_h_outlier])
    arr[2] = np.array([count_h_mean, ntop+count_h_mean])
    arr[3] = np.array([count_h_sc, ntop+count_h_sc])
    
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

    
    