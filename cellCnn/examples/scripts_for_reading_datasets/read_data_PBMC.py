import os, glob, pickle, fcm
import numpy as np
import pandas as pd
import cellCnn
from cellCnn.utils import ftrans, mkdir_p
from  sklearn.preprocessing import StandardScaler

FCS_DATA_PATH = '/Volumes/biol_imsb_claassen_s1/eiriniar/data/bodenmiller_PBMC'


def flat_list(big_list):
	return [item for sublist in big_list for item in sublist]

def get_immediate_subdirectories(a_dir):
	return [name for name in os.listdir(a_dir)
			if os.path.isdir(os.path.join(a_dir, name))]
			
def no_inhibitor_lookup_full(data_path, stimuli, ctypes, marker_idx):
	lookup = dict()
	dose = 'H'
	labels = None
	print stimuli
	
	for key in get_immediate_subdirectories(data_path):
		subdir = os.path.join(data_path, key)
		full_data_list = []
		stim_list, ctype_list = [], []
	
		scaler = StandardScaler(with_std=False)
		for ii, ctype in enumerate(ctypes):
			for jj, stim in enumerate(stimuli):
				
				tu = (key, ctype, dose, stim)
				fname = os.path.join(subdir, '{0}_{1}_{2}{3}.fcs'.format(*tu))
				try:

					# read the .fcs file
					x_full = fcm.loadFCS(fname, transform=None)
					if labels is None:
						labels = [x_full.channels[ii] for ii in marker_idx]

					# keep only interesting markers and arcsinh-transform the data
					x_full = ftrans(np.asarray(x_full)[:,marker_idx], 5)
						
					# fit a mean-shift scaler on control CD4+ T-cells (only on intracellular markers)
					if (ctype == 'cd4+') and (stim == '05'):
						scaler.fit(x_full[:,10:])
						
					# and transform all cell types using this scaler
					x_full[:,10:] = scaler.transform(x_full[:,10:])
						
					# accumulate all the data seen so far along with their labels
					full_data_list.append(x_full)
					stim_list.append(jj * np.ones(x_full.shape[0], dtype=int))
					ctype_list.append([ctype] * x_full.shape[0])

				except Exception: 
					pass

		lookup[key] = {'X': np.vstack(full_data_list),
					   'y': np.hstack(stim_list),
					   'ctype' : flat_list(ctype_list),
					   'labels' : labels,
					   'scaler' : scaler}
	return lookup


def main():

	# cell types manually gated
	CTYPES = ['cd4+', 'cd8+', 'cd14+hladrmid', 'cd14-hladrmid', 'cd14+surf-', 'cd14-surf-',
			 'dendritic', 'igm+', 'igm-', 'nk']
	
	# channels measured in this experiment
	CH = ['Time', 'Cell_length', 'CD3', 'CD45', 'BC1', 'BC2', 'pNFkB',
			'pp38', 'CD4', 'BC3', 'CD20', 'CD33', 'pStat5', 'CD123',
			'pAkt', 'pStat1', 'pSHP2', 'pZap70', 'pStat3', 'BC4', 'CD14',
			'pSlp76', 'BC5', 'pBtk', 'pPlcg2', 'pErk', 'BC6', 'pLat',
			'IgM', 'pS6', 'HLA-DR', 'BC7', 'CD7', 'DNA-1', 'DNA-2']
	
	# intracelluler makrers
	PH_LABELS = ['pStat1', 'pStat3', 'pStat5', 'pNFkB', 'pp38', 'pAkt', 'pSHP2', 'pZap70',
				'pSlp76', 'pBtk', 'pPlcg2', 'pErk', 'pLat', 'pS6']
	PH_IDX = [CH.index(label) for label in PH_LABELS]

	# cell surface markers
	CD_LABELS = ['CD45', 'CD3', 'CD4', 'CD7', 'CD20', 'IgM', 'CD33',
				 'CD14', 'HLA-DR', 'CD123']
	CD_IDX = [CH.index(label) for label in CD_LABELS]

	# all interesting markers that should be read
	labels = CD_LABELS + PH_LABELS
	marker_idx = CD_IDX + PH_IDX

	# different stimuli considered in this experiemnt
	STIMULI = ['02', '03', '04', '06', '07', '08', '09', '10', '11', '12', '01']
	STIM_NAMES = ['IL-3', 'IL-2', 'IL-12', 'G-CSF', 'GM-CSF',
				  'BCR', 'IFN-g', 'IFN-a', 'LPS', 'PMA', 'Vanadate']


	# store the pre-processed datasets
	pickle_dir = os.path.join(cellCnn.__path__[0], 'examples', 'data')
	mkdir_p(pickle_dir)
	
	for (s_code, s_name) in zip(STIMULI, STIM_NAMES):
		pickle_file = os.path.join(pickle_dir, s_name + '_vs_control.pkl')
		lookup = no_inhibitor_lookup_full(data_path=FCS_DATA_PATH,
										 stimuli=['05', s_code],
										 ctypes=CTYPES,
										 marker_idx=marker_idx)

		with open(pickle_file, 'wb') as f:
			pickle.dump(lookup, f, -1)

	return 0


if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		sys.stderr.write("User interrupt!\n")
		sys.exit(-1)
