import os, glob, pickle, fcm
import numpy as np
import pandas as pd
import cellCnn
from cellCnn.utils import ftrans, mkdir_p, logrank_pval
from cellCnn.plotting import plot_KM
from  sklearn.preprocessing import StandardScaler

BASEDIR = '/Volumes/biol_imsb_claassen_s1/eiriniar/Data/HIV_cohort'
PATH_COMP =  os.path.join(BASEDIR, 'HIV_compensated_gated_live_T_cells')
prefix = 'export'
suffix = 'liveCD3+.fcs'

# markers kept for further analysis
labels = ['KI67', 'CD127', 'CCR7', 'CD27', 'CCR5', 'CD45RO', 'CD28', 'CD57', 'CD4', 'CD8']
marker_idx = range(3, 7) + range(8, 11) + range(13, 16)


def load_data():
	infofile = os.path.join(BASEDIR, 'clinical_data_flow_repository.csv')
	df = pd.read_csv(infofile, sep='\t')
	data_list = []
	name_list = []
	ytime_l, ystatus_l, id_l = [], [], []
		
	for ii in df.index: 
		ID, y1, y2 = df.iloc[ii]
		
		# analyze only samples with positive survival times
		if (y1 > 0):

			file1 = os.path.join(PATH_COMP, '_'.join([prefix, str(ID), suffix]))
			
			try:
				# load the raw .fcs data
				X = fcm.loadFCS(file1, transform=None)

				# keep only interesting markers and arcsinh-transform the data
				X = ftrans(np.asarray(X)[:,marker_idx], 150)
			   
				# discard samples with less than 3000 cells
				if X.shape[0] > 3000:
					print X.shape
					data_list.append(X)
					name_list.append(file1)
					ytime_l.append(y1)
					ystatus_l.append(y2)
					id_l.append(ii)
	
			except Exception:
				print 'Could not find or load sample: ' + file1
				pass

	y = np.hstack([np.hstack(ytime_l).reshape(-1,1),
				   np.hstack(ystatus_l).reshape(-1,1),
				   np.hstack(id_l).reshape(-1,1)])
	return data_list, name_list, y


def main():

	# load the data
	data_list, name_list, y = load_data()
	
	# and save them in pickle format
	save_path = os.path.join(cellCnn.__path__[0], 'examples', 'data', 'HIV.pkl')
	D = {'data':data_list, 'y':y, 'labels':labels, 'filenames':name_list}
	with open(save_path, 'wb') as f:
		pickle.dump(D, f, -1)


	'''
	# event occurence list    
	occurred = [x for i, x in enumerate(data_list) if y[i,1] == 1]
	not_occurred = [x for i, x in enumerate(data_list) if y[i,1] == 0]
	y1 = y[y[:,1] == 1]
	y0 = y[y[:,1] == 0]
	
	# split the examples randomly into a training (2/3) and test (1/3) cohort
	# both cohorts should contain equal percentage of cencored data
	sep1 = len(y1) / 3
	sep0 = len(y0) / 3

	# randomly partition samples into training and test cohort
	# use the same partition for analysis with Citrus
	file_occurred = [fname for i, fname in enumerate(name_list) if y[i,1] == 1]
	file_not_occurred = [fname for i, fname in enumerate(name_list) if y[i,1] == 0]
	
	# train samples
	train_files = [fname.split('/')[-1] for fname in (file_occurred[sep1:] + file_not_occurred[sep0:])]
	pd.DataFrame(train_files).to_csv(os.path.join(BASEDIR, 'CITRUS_training_files.csv'), index=False)

	# test files
	test_files = [fname.split('/')[-1] for fname in (file_occurred[:sep1] + file_not_occurred[:sep0])]
	pd.DataFrame(test_files).to_csv(os.path.join(BASEDIR, 'CITRUS_test_files.csv'), index=False)

	# get indices for sorting the test files and their labels according to sample id
	# (initially all uncensored samples come first, then all censored)
	# we need this sorting so that the order of Citrus predictions is matched
	test_file_ids = np.array([int(name.split('_')[1]) for name in test_files])
	test_idx_sorted = np.argsort(test_file_ids)

	# plot results pre-computed with Citrus
	WDIR = os.path.join(cellCnn.__path__[0], 'examples')
	OUTDIR = os.path.join(WDIR, 'output', 'HIV')
	mkdir_p(OUTDIR)

	CITRUS_PRED_PATH = os.path.join(BASEDIR, 'CITRUS_HIV_test_cohort_output', 'euler_test_predictions.csv')
    risk_citrus = np.asarray(pd.read_csv(CITRUS_PRED_PATH, sep=',')['s0'])
    g2 = np.squeeze(risk_citrus > np.median(risk_citrus))
    stime_sorted, censor_sorted = y_valid[test_idx_sorted,0], y_valid[test_idx_sorted,1]
    citrus_pval_v = logrank_pval(stime_sorted, censor_sorted, g2)
    fig_c = os.path.join(OUTDIR, 'citrus_cox_test.eps')
    plot_KM(stime_sorted, censor_sorted, g2, citrus_pval_v, fig_c)
    '''


if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		sys.stderr.write("User interrupt!\n")
		sys.exit(-1)

