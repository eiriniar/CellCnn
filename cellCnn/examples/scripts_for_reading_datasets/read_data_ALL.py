
import os, glob, pickle
import numpy as np
import pandas as pd
import cellCnn
from cellCnn.utils import ftrans, mkdir_p


def main():

	PATH = "/Volumes/biol_imsb_claassen_s1/eiriniar/Data/viSNE/mrd_debarcode"
	mrd_file = os.path.join(PATH, 'mrd_debarcoded.csv')
	healthy_file = os.path.join(PATH, 'healthy_debarcoded.csv')
	control_file = os.path.join(PATH, 'visne_marrow1.csv')
	mrd_data = pd.read_csv(mrd_file, sep=',')
	healthy_data = pd.read_csv(healthy_file, sep=',')
	control_data = pd.read_csv(control_file, sep=',')

	# all available channels
	channels = list(control_data.columns)

	# which markers should be kept for further analysis
	full_labels = ['CD19(Nd142)Di','CD22(Nd143)Di', 'CD47(Nd145)Di','CD79b(Nd146)Di',
					'CD20(Sm147)Di', 'CD34(Nd148)Di','CD179a(Sm149)Di','CD72(Eu151)Di',
					'IgM-i(Eu153)Di','CD45(Sm154)Di','CD10(Gd156)Di',
					'CD179b(Gd158)Di','CD11c(Tb159)Di','CD14(Gd160)Di','CD24(Dy161)Di',
					'CD127(Dy162)Di','TdT(Dy163)Di','CD15(Dy164)Di','Pax5(Ho165)Di',
					'CD38(Er168)Di','CD3(Er170)Di','CD117(Yb171)Di',
					'CD49d(Yb172)Di','CD33(Yb173)Di','HLADR(Yb174)Di','IgM-s(Lu175)Di',
					'CD7(Yb176)Di']

	labels = [label.split('(')[0] for label in full_labels]

	# which columns correspond to the interesting markers
	marker_idx = [channels.index(label) for label in full_labels]

	# keep only interesting markers and arcsinh-transform the data
	x_mrd = ftrans(np.asarray(mrd_data)[:,marker_idx], 5)
	x_healthy = ftrans(np.asarray(healthy_data)[:,marker_idx], 5)
	x_control = ftrans(np.asarray(control_data)[:,marker_idx], 5)

	# select CD10+ blasts
	cd10_idx = np.argsort(x_mrd[:,10])
	x_mrd = x_mrd[cd10_idx[-500:]]
	
	# save the pre-processed dataset
	pickle_dir = os.path.join(cellCnn.__path__[0], 'examples', 'data')
	mkdir_p(pickle_dir)
	pickle_file = os.path.join(pickle_dir, 'ALL.pkl')
	
	data_dict = {'control': x_control,
				 'healthy': x_healthy,
				 'ALL': x_mrd,
				 'labels': labels}
	with open(pickle_file, 'wb') as f:
			pickle.dump(data_dict, f, -1)

	return 0


# combine several .csv files (debarcoded subpopulations) into a big one 
def combine_csv(flist, labels, savepath):
	xlist = []
	for i, filename in enumerate(flist):
		data = np.asarray(pd.read_csv(filename, sep=','))
		xlist.append(data)
	data = np.vstack(xlist)
	pd.DataFrame(data, columns=labels).to_csv(savepath, sep=',', index=False)



if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		sys.stderr.write("User interrupt!\n")
		sys.exit(-1)


