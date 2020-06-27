""" Copyright 2016-2017 ETH Zurich, Eirini Arvaniti and Manfred Claassen.

This module supports a CellCnn analysis from the command line.

"""

import argparse
import os
import sys
import pickle
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, KFold
from cellCnn.utils import get_data, save_results, mkdir_p, get_selected_cells
from cellCnn.plotting import plot_results, plot_filters, discriminative_filters
from cellCnn.model import CellCnn

logging.basicConfig(
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    # IO-specific
    parser.add_argument('-f', '--fcs', required=True,
                        help='file specifying the FCS file names and corresponding labels')
    parser.add_argument('-m', '--markers', required=True,
                        help='file specifying the names of markers to be used for analysis')
    parser.add_argument('-i', '--indir', default='./',
                        help='directory where input FCS files are located')
    parser.add_argument('-o', '--outdir', default='output',
                        help='directory where output will be generated')
    parser.add_argument('-p', '--plot', action='store_true', default=True,
                        help='whether to plot results ')
    parser.add_argument('--export_selected_cells', action='store_true', default=False,
                        help='whether to export selected cell populations')
    parser.add_argument('--export_csv', action='store_true', default=False,
                        help='whether to export network weights as csv files')
    parser.add_argument('-l', '--load_results', action='store_true', default=False,
                        help='whether to load precomputed results')

    # data preprocessing
    parser.add_argument('--train_perc', type=float, default=0.75,
                        help='percentage of samples to be used for training')
    parser.add_argument('--arcsinh', dest='arcsinh', action='store_true',
                        help='preprocess the data with arcsinh')
    parser.add_argument('--no_arcsinh', dest='arcsinh', action='store_false',
                        help='do not preprocess the data with arcsinh')
    parser.set_defaults(arcsinh=True)
    parser.add_argument('--cofactor', type=int, default=5,
                        help='cofactor for the arcsinh transform')
    parser.add_argument('--scale', dest='scale', action='store_true',
                        help='z-transform features (mean=0, std=1) prior to training')
    parser.add_argument('--no_scale', dest='scale', action='store_false',
                        help='do not z-transform features (mean=0, std=1) prior to training')
    parser.set_defaults(scale=True)
    parser.add_argument('--quant_normed', action='store_true', default=False,
                        help='only use this option if the input data already lies in the [0, 1] interval,'
                             'e.g. after quantile normalization')

    # multi-cell input specific
    parser.add_argument('--ncell', type=int, help='number of cells per multi-cell input',
                        default=200)
    parser.add_argument('--nsubset', type=int, help='number of multi-cell inputs',
                        default=1000)
    parser.add_argument('--per_sample', action='store_true', default=False,
                        help='whether nsubset refers to each class or each sample')
    parser.add_argument('--subset_selection', choices=['random', 'outlier'], default='random',
                        help='generate random or outlier-enriched multi-cell inputs')

    # neural network specific
    parser.add_argument('--maxpool_percentages', nargs='+', type=float,
                        help='list of choices (percentage of multi-cell input) for top-k max pooling',
                        default=[0.01, 1, 5, 20])
    parser.add_argument('--nfilter_choice', nargs='+', type=int,
                        help='list of choices for number of filters', default=list(range(3, 10)))
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='learning rate for the Adam optimization algorithm')
    parser.add_argument('--coeff_l1', type=float, default=0,
                        help='coefficient for L1 weight regularization')
    parser.add_argument('--coeff_l2', type=float, default=0.01,
                        help='coefficient for L2 weight regularization')
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='maximum number of iterations through the data')
    parser.add_argument('--patience', type=int, default=5,
                        help='number of epochs before early stopping')

    # analysis specific
    parser.add_argument('--seed', type=int, default=12321,
                        help='random seed')
    parser.add_argument('--nrun', type=int, default=15,
                        help='number of neural network configurations to try (should be >= 3)')
    parser.add_argument('--regression', action='store_true', default=False,
                        help='whether it is a regression problem (default is classification)')
    parser.add_argument('--dendrogram_cutoff', type=float, default=.4,
                        help='cutoff for hierarchical clustering of filter weights')
    parser.add_argument('--accur_thres', type=float, default=.9,
                        help='keep filters from models achieving at least this accuracy '
                             ' (or at least from the best 3 models)')
    parser.add_argument('-v', '--verbose', type=int, choices=[0, 1], default=1,
                        help='output verbosity')

    # plot specific
    parser.add_argument('--filter_diff_thres', type=float, default=0.2,
                        help='threshold that defines which filters are discriminative')
    parser.add_argument('--filter_response_thres', type=float, default=0,
                        help='threshold that defines the selected cell population per filter')
    parser.add_argument('--stat_test', choices=[None, 'ttest', 'mannwhitneyu'],
                        help='statistical test for comparing cell population frequencies of two '
                             'groups of samples')
    parser.add_argument('--group_a', default='group A',
                        help='name of the first class')
    parser.add_argument('--group_b', default='group B',
                        help='name of the second class')
    parser.add_argument('--group_names', nargs='+', default=None,
                        help='list of class names')
    parser.add_argument('--tsne_ncell', type=int, help='number of cells to include in t-SNE maps',
                        default=10000)
    args = parser.parse_args()

    # read in the data
    fcs_info = np.array(pd.read_csv(args.fcs, sep=','))
    marker_names = list(pd.read_csv(args.markers, sep=',').columns)
    # if the samples have already been pre-processed via quantile normalization
    # we should not perform arcsinh transformation
    if args.quant_normed:
        args.arcsinh = False
    samples, phenotypes = get_data(args.indir, fcs_info, marker_names,
                                   args.arcsinh, args.cofactor)

    # set random seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # generate training/validation sets
    val_perc = 1 - args.train_perc
    n_splits = int(1. / val_perc)
    # stratified CV for classification problems
    if not args.regression:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    # simple CV for regression problems
    else:
        skf = KFold(n_splits=n_splits, shuffle=True)
    train, val = next(skf.split(np.zeros((len(phenotypes), 1)), phenotypes))

    train_samples = [samples[i] for i in train]
    valid_samples = [samples[i] for i in val]
    train_phenotypes = [phenotypes[i] for i in train]
    valid_phenotypes = [phenotypes[i] for i in val]

    logger.info(f"Samples used for model training: {[fcs_info[i][0] for i in train]}")
    logger.info(f"Samples used for validation: {[fcs_info[i][0] for i in val]}")

    # always generate multi-cell inputs on a per-sample basis for regression
    if args.regression:
        args.per_sample = True

    if not args.load_results:
        # run CellCnn
        model = CellCnn(ncell=args.ncell,
                        nsubset=args.nsubset,
                        per_sample=args.per_sample,
                        subset_selection=args.subset_selection,
                        scale=args.scale,
                        quant_normed=args.quant_normed,
                        maxpool_percentages=args.maxpool_percentages,
                        nfilter_choice=args.nfilter_choice,
                        nrun=args.nrun,
                        regression=args.regression,
                        learning_rate=args.learning_rate,
                        coeff_l1=args.coeff_l1,
                        coeff_l2=args.coeff_l2,
                        max_epochs=args.max_epochs,
                        patience=args.patience,
                        dendrogram_cutoff=args.dendrogram_cutoff,
                        accur_thres=args.accur_thres,
                        verbose=args.verbose)
        model.fit(train_samples=train_samples, train_phenotypes=train_phenotypes,
                  valid_samples=valid_samples, valid_phenotypes=valid_phenotypes,
                  outdir=args.outdir)
        # save results for subsequent analysis
        results = model.results
        pickle.dump(results, open(os.path.join(args.outdir, 'results.pkl'), 'wb'))
    else:
        results = pickle.load(open(os.path.join(args.outdir, 'results.pkl'), 'rb'))

    if args.export_csv:
        save_results(results, args.outdir, marker_names)

    # plot results
    if args.plot or args.export_selected_cells:
        plotdir = os.path.join(args.outdir, 'plots')
        plot_filters(results, marker_names, os.path.join(plotdir, 'filter_plots'))
        _v = discriminative_filters(results, os.path.join(plotdir, 'filter_plots'),
                                    filter_diff_thres=args.filter_diff_thres,
                                    show_filters=True)
        filter_info = plot_results(results, train_samples, train_phenotypes,
                                   marker_names, os.path.join(plotdir, 'training_plots'),
                                   filter_diff_thres=args.filter_diff_thres,
                                   filter_response_thres=args.filter_response_thres,
                                   stat_test=args.stat_test,
                                   group_a=args.group_a, group_b=args.group_b,
                                   group_names=args.group_names,
                                   tsne_ncell=args.tsne_ncell,
                                   regression=args.regression,
                                   show_filters=False)
        _v = plot_results(results, valid_samples, valid_phenotypes,
                          marker_names, os.path.join(plotdir, 'validation_plots'),
                          filter_diff_thres=args.filter_diff_thres,
                          filter_response_thres=args.filter_response_thres,
                          stat_test=args.stat_test,
                          group_a=args.group_a, group_b=args.group_b,
                          group_names=args.group_names,
                          tsne_ncell=args.tsne_ncell,
                          regression=args.regression,
                          show_filters=False)
        if args.export_selected_cells:
            csv_dir = os.path.join(args.outdir, 'selected_cells')
            mkdir_p(csv_dir)
            nfilter = len(filter_info)
            sample_names = [name.split('.fcs')[0] for name in list(fcs_info[:, 0])]
            # for each sample
            for x, x_name in zip(samples, sample_names):
                flags = np.zeros((x.shape[0], 2 * nfilter))
                columns = []
                # for each filter
                for i, (filter_idx, thres) in enumerate(filter_info):
                    flags[:, 2 * i:2 * (i + 1)] = get_selected_cells(
                        results['selected_filters'][filter_idx], x, results['scaler'], thres, True)
                    columns += ['filter_%d_continuous' % filter_idx, 'filter_%d_binary' % filter_idx]
                df = pd.DataFrame(flags, columns=columns)
                df.to_csv(os.path.join(csv_dir, x_name + '_selected_cells.csv'), index=False)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("User interrupt!\n")
        sys.exit(-1)
