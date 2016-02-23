
import os, glob, pickle, fcm
import numpy as np
import pandas as pd
import cellCnn
from cellCnn.utils import ftrans, mkdir_p


def read_healthy_data(basedir, keys, stimuli, marker_idx):
	lookup = dict()

	for key in keys:
		subdir = os.path.join(basedir, key)
		data_list, stim_list = [], []
	
		for jj, stim in enumerate(stimuli):
			fname = os.path.join(subdir,
								'_'.join([key, stim, 'PhenoGraph.csv']))
			try:

				# load the raw data
				x = fcm.loadFCS(fname, transform=None)
				print [x.channels[ii] for ii in marker_idx]
				
				# select interesting markers and arcsinh-transform
				x = ftrans(np.asarray(x)[:,marker_idx], 5)
				
				# merge data from different stimulation conditions
				data_list.append(x)
				
			except Exception:
				print 'Problem loading: ' + fname 
				pass

		lookup[key] = np.vstack(data_list) 
	return lookup


def main():

	# stimulation conditions in this experiment
	STIM = ['Basal1', 'Basal2', 'AICAR', 'Flt3L', 'G-CSF', 'GM-CSF', 'IFNa', 'IFNg',
		'IL-10', 'IL-27', 'IL-3', 'IL-6', 'PMAiono', 'PVO4', 'SCF', 'TNFa', 'TPO']
	full_stim_names = ['_'.join(['NoDrug', stim]) for stim in STIM] + ['BEZ-235_Basal1']
	
	# all available channels in this experiment
	channels = ['Time','Cell_length','DNA1','DNA2','BC1','BC2','BC3','BC4','BC5','BC6',
				'pPLCg2','CD19','p4EBP1','CD11b','pAMPK','pSTAT3','CD34','pSTAT5',
				'pS6','pCREB','pc-Cbl','CD45','CD123','pSTAT1','pZap70-Syk','CD33',
				'CD47','pAKT','CD7','CD15','pRb','CD44','CD38','pErk1-2','CD3',
				'pP38','CD117','cCaspase3','HLA-DR','CD64','CD41',
				'Viability','PhenoGraph']

	# which markers should be kept for further analysis
	labels = [  'CD19', 'CD11b', 'CD34', 'CD45','CD123','CD33',
				'CD47', 'CD7', 'CD15', 'CD44','CD38','CD3',
				'CD117', 'HLA-DR','CD64','CD41']

	# which columns correspond to the interesting markers
	marker_idx = [channels.index(label) for label in labels]

	# data directory
	FCS_DATA_PATH = '/Volumes/biol_imsb_claassen_s1/eiriniar/Data/phenograph_data'

	# read the data from healthy samples
	healthy_keys = ['H' + str(i) for i in range(1,6)]
	D = read_healthy_data(FCS_DATA_PATH, healthy_keys, full_stim_names, marker_idx)
	aml_dict = {'healthy_BM': [(key, D[key]) for key in ['H1', 'H2', 'H3','H5','H4']]}

	# map .txt files back to patient identifiers
	mapping = { 0:'SJ10', 2:'SJ12', 3:'SJ13', 4:'SJ14', 5:'SJ15',
				6:'SJ16', 8:'SJ1', 9:'SJ1', 10:'SJ2', 11:'SJ2',
				12:'SJ3', 13:'SJ3', 14:'SJ4', 15:'SJ5', 17:'SJ7'}
	
	# read the data from AML samples
	# gated blast populations were downloaded as .txt files from Cytobank
	# CAREFUL when reading these .txt files:
	# they are tab-separated and include an extra first column (cell index)
	AML_files = glob.glob(os.path.join(FCS_DATA_PATH, 'AML_blasts', '*.txt'))
   
	# only include patients with sufficiently high blast counts 
	# (>10% of total cell counts)
	for sj in [0,2,3,4,5,6,8,9,10,11,12,13,14,15,17]:
		fname = AML_files[sj]
		t = pd.read_csv(fname, skiprows=1, sep='\t', index_col=0)
		print [list(t.columns)[ii] for ii in marker_idx]
		data_blasts = ftrans(np.asarray(t)[:, marker_idx], 5)
		if mapping[sj] not in aml_dict:
			aml_dict[mapping[sj]] = data_blasts

	
	# save the pre-processed dataset
	pickle_dir = os.path.join(cellCnn.__path__[0], 'examples', 'data')
	mkdir_p(pickle_dir)
	pickle_file = os.path.join(pickle_dir, 'AML.pkl')
	aml_dict['labels'] = labels
	with open(pickle_file, 'wb') as f:
		pickle.dump(aml_dict, f, -1)

	return 0


if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		sys.stderr.write("User interrupt!\n")
		sys.exit(-1)


