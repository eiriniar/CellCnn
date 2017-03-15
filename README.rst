
=======
CellCnn
=======

Installation 
============

CellCnn is written in Python2.7. After having Python2.7 running on your system, please do the following:

1. Clone the CellCnn repository:
    ``git clone https://github.com/eiriniar/CellCnn.git``

2. Install the CellCnn dependencies:
    ``pip install -r https://raw.githubusercontent.com/eiriniar/CellCnn/master/requirements.txt``

3. To install CellCnn, run the following command after replacing ``path_to_CellCnn`` with the actual path in your system:
    ``pip install -e path_to_CellCnn/CellCnn``

----

For the analysis of mass/flow cytometry data, the package ``fcm`` needs to be additionally installed:

``pip install fcm==0.9.1``

The ``fcm`` package internally uses a deprecated matplotlib function which raises an exception.
As a workaround, you can get ``fcm`` running by editing the source code in ``path_to_python/lib/python2.7/site-packages/fcm/core/gate.py`` (``path_to_python`` has to be replaced with the actual path in your system). Specifically:

1. Comment out line: ``from matplotlib.nxutils import points_inside_poly``
2. Uncomment line: ``idxs = points_in_poly(self.vert, fcm.view()[:, chan])``
3. Comment out line: ``idxs = points_inside_poly(fcm.view()[:, chan], self.vert)``


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

``python ../run_analysis.py -f NK_fcs_samples_with_labels.csv -m NK_markers.csv -i NK_cell_dataset/gated_NK/ -o outdir_NK --max_epochs 15 --nrun 10 --train_perc 0.6 --ncell_pooled 5 10 --plot --export_csv --group_a CMV- --group_b CMV+``

The above command performs a binary classification CellCnn analysis, exports the learned filter weights as CSV files in the directory ``outdir_NK/csv_results`` and generates some result plots in ``outdir_NK/plots``. After
performing model training once, you can refine the plots with different cutoff values for the selected filters and
cell populations. Training does not have to be repeated for refining the plots. The pre-computed results can be used with the option ``--load_results``.

Another relevant argument is ``--export_selected_cells``, which produces a CSV result file for each input FCS file and stores it in ``outdir/selected_cells``. Rows in the CSV result file correspond to cells in the order
found in the FCS input file.
The CSV result file contains two columns per selected filter, the first indicating the cell filter response as a continuous value and the second containing a binary value resulting from thresholding the continuous cell filter response. This later column is an indicator of whether a cell belongs to the cell population selected by a particular filter.

``python ../run_analysis.py -f NK_fcs_samples_with_labels.csv -m NK_markers.csv -i NK_cell_dataset/gated_NK/ -o outdir_NK --plot --group_a CMV- --group_b CMV+ --filter_response_thres 0.3 --load_results --export_selected_cells``


Documentation
=============

For additional information, CellCnn's documentation is hosted on http://eiriniar.github.io/CellCnn/
