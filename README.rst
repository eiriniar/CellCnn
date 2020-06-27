
=======
CellCnn
=======

Installation
============

This branch contains a Python 3.7 implementation of CellCnn. We recommend using it with
`pipenv <https://pypi.org/project/pipenv/>`_, by following the steps below:

#. Clone the CellCnn repository:
    ``git clone https://github.com/eiriniar/CellCnn.git``

#. Go to the CellCnn root directory:
    ``cd CellCnn``

#. Install CellCnn and its dependencies:
    ``pipenv install '-e .'``

The above steps have to be performed only once.
Then, each time you want to perform a CellCnn analysis, go to the CellCnn root directory and activate the pipenv virtual environment by running:

#.   ``pipenv shell``


Usage
=====

Examples are provided in the subfolder ``CellCnn/cellCnn/examples``.

----

Alternatively, for the analysis of mass/flow cytometry samples, CellCnn can be run from the command line.
To get a list of command line options please run:

``python run_analysis.py --help``

For a CellCnn analysis with default settings only two arguments have to be provided:

``python run_analysis.py -f fcs_samples_with_labels.csv -m markers.csv`` 

| The first input argument is a two-column CSV file, where the first column specifies input sample filenames and the second column the corresponding class labels. An example file is provided in ``CellCnn/cellCnn/examples/NK_fcs_samples_with_labels.csv``.
| The second input argument is a CSV file containing the names of markers/channels that should be used for the analysis. An example file is provided in ``CellCnn/cellCnn/examples/NK_markers.csv``.

For example, to perform the analysis outlined in ``CellCnn/cellCnn/examples/NK_cell.ipynb``
from the command line, you can run the following (assuming your current directory is ``CellCnn/cellCnn/examples``):

``python ../run_analysis.py -f NK_fcs_samples_with_labels.csv -m NK_markers.csv -i NK_cell_dataset/gated_NK/ -o outdir_NK --export_csv --group_a CMV- --group_b CMV+ --verbose 0``

The above command performs a binary classification CellCnn analysis, exports the learned filter weights as CSV files in the directory ``outdir_NK/exported_filter_weights`` and generates result plots in ``outdir_NK/plots``. The following plots are generated:

filter_plots
""""""""""""

- clustered_filter_weights.pdf :
    Filter weight vectors from all trained networks that pass a validation accuracy
    threshold, grouped in clusters via hierarchical clustering. Each row corresponds to
    a filter. The last column(s) indicate the weight(s) connecting each filter to the output
    class(es). Indices on the y-axis indicate the filter cluster memberships, as a
    result of the hierarchical clustering procedure.
- consensus_filter_weights.pdf :
    One representative filter per cluster is chosen (the filter with minimum distance to all
    other memebers of the cluster). We call these selected filters "consensus filters".
- best_net_weights.pdf :
    Filter weight vectors of the network that achieved the highest validation accuracy.
- filter_response_differences.pdf :
    Difference in cell filter response between classes for each consensus filter.
    To compute this difference for a filter, we first choose a filter-specific class, that's
    the class with highest output weight connection to the filter. Then we compute the
    average cell filter response (value after the pooling layer) for validation samples
    belonging to the filter-specific class (``v1``) and the average cell filter response
    for validation samples not belonging to the filter-specific class (``v0``).
    The difference is computed as ``v1 - v0``. For regression problems, we cannot compute
    a difference between classes. Instead we compute Kendall's rank correlation coefficient
    between the predictions of each individual filter (value after the pooling layer) and
    the true response values. This plot helps decide on a cutoff (``filter_diff_thres`` parameter)
    for selecting discriminative filters.

training_plots
""""""""""""""

These plots are generated on the basis of samples used for model training.

- tsne_all_cells.png :
    Marker distribution overlaid on t-SNE map.

In addition, the following plots are produced for each selected filter (e.g. filter ``i``):

- cdf_filter_i.pdf :
    Cumulative distribution function of cell filter response for filter ``i``. This plot
    helps decide on a cutoff (``filter_response_thres`` parameter) for selecting the
    responding cell population.
- selected_population_distribution_filter_i.pdf :
    Histograms of univariate marker expression profiles for the cell population selected by
    filter ``i`` vs all cells.
- selected_population_frequencies_filter_i.pdf :
    Boxplot of selected cell population frequencies in samples of the different classes, if running a classification problem.
    For regression settings, a scatter plot of selected cell population frequencies vs response variable is generated.
- tsne_cell_response_filter_i.png :
    Cell filter response overlaid on t-SNE map.
- tsne_selected_cells_filter_i.png :
    Marker distribution of selected cell population overlaid on t-SNE map.

validation_plots
""""""""""""""""

Same as the training_plots, but generated on the basis of samples used for model validation.

----

After performing model training once, you can refine the plots with different cutoff values for the selected filters and
cell populations. Training does not have to be repeated for refining the plots. The pre-computed results can be used with the option ``--load_results``.

Another relevant argument is ``--export_selected_cells``, which produces a CSV result file for each input FCS file and stores it in ``outdir/selected_cells``. Rows in the CSV result file correspond to cells in the order found in the FCS input file.
The CSV result file contains two columns per selected filter, the first indicating the cell filter response as a continuous value and the second containing a binary value resulting from thresholding the continuous cell filter response. This later column is an indicator of whether a cell belongs to the cell population selected by a particular filter.

``python ../run_analysis.py -f NK_fcs_samples_with_labels.csv -m NK_markers.csv -i NK_cell_dataset/gated_NK/ -o outdir_NK --group_a CMV- --group_b CMV+ --filter_response_thres 0.3 --load_results --export_selected_cells``


Documentation
=============

For additional information, CellCnn's documentation is hosted on http://eiriniar.github.io/CellCnn/
