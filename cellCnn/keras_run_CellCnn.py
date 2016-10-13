import sys, os
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
import copy

from cellCnn.utils import combine_samples, landmark_normalization, normalize_outliers_to_control, normalize_outliers
from cellCnn.utils import compute_consensus_profiles, keras_param_vector
from cellCnn.utils import generate_subsets, generate_biased_subsets
from cellCnn.downsample import knn_dist, knn_dist_memory_optimized
from cellCnn.theano_utils import select_top
from cellCnn.theano_utils import float32, int32, activity_KL

import theano
import theano.tensor as T
from keras.layers import Input, Dense, Flatten, Reshape, Lambda, Activation, Dropout
from keras.layers.convolutional import Convolution1D, MaxPooling1D, AveragePooling1D
from keras.layers.noise import GaussianNoise
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l1l2, activity_l1
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau


def train_model(train_samples, train_phenotypes, labels,
				valid_samples=None, valid_phenotypes=None, generate_valid_set=True,
				train_sample_flags=None, valid_sample_flags=None, 
				landmark_norm=None, scale=True, 
				ncell=500, nsubset=4096, subset_selection='random', nrun=10,
				pooling='max', ncell_pooled=[1,3,5], regression=False, nfilter_choice=[2,3,4,5],
				learning_rate=None, coeff_l1=0, coeff_l2=1e-4, dropout=False,
				coeff_activity=0, max_epochs=50, verbose=1,
				mode='supervised', patience=10,
				select_filters='consensus', dendrogram_cutoff=0.4,
				accur_thres=.95, outdir=None):
	
	'''
	train_samples: list with input samples (e.g. cytometry samples) as numpy arrays
	train_phenotype: list of phenotypes associated with the samples in train_samples
	labels: names of measured markers in train_samples
	'''
	
	# copy the list of samples so that they are not modified in place
	train_samples = copy.deepcopy(train_samples)
	if valid_samples is not None:
		valid_samples = copy.deepcopy(valid_samples)
		
	# create dummy single-cell flags if not given
	if train_sample_flags is None:
		train_sample_flags = [np.zeros((x.shape[0],1), dtype=int) for x in train_samples]
	if (valid_samples is not None) and (valid_sample_flags is None):
		valid_sample_flags = [np.zeros((x.shape[0],1), dtype=int) for x in valid_samples]

	if landmark_norm is not None:
		idx_to_normalize = [labels.index(label) for label in landmark_norm]
		train_samples = landmark_normalization(train_samples, idx_to_normalize)
	
		if valid_samples is not None:
			valid_samples = landmark_normalization(valid_samples, idx_to_normalize)
			   
	# normalize extreme values
	# we assume that 0 corresponds to the control class
	if subset_selection == 'outlier':
		ctrl_list = [train_samples[i] for i in np.where(np.array(train_phenotypes) == 0)[0]]
		test_list = [train_samples[i] for i in np.where(np.array(train_phenotypes) != 0)[0]]
		train_samples = normalize_outliers_to_control(ctrl_list, test_list)

		if valid_samples is not None:
			ctrl_list = [valid_samples[i] for i in np.where(np.array(valid_phenotypes) == 0)[0]]
			test_list = [valid_samples[i] for i in np.where(np.array(valid_phenotypes) != 0)[0]]
			valid_samples = normalize_outliers_to_control(ctrl_list, test_list)

	if (valid_samples is None) and (not generate_valid_set):
		sample_ids = range(len(train_phenotypes))
		X_train, id_train, z_train = combine_samples(train_samples, sample_ids, train_sample_flags)
		
	elif (valid_samples is None) and generate_valid_set:
		sample_ids = range(len(train_phenotypes))
		X, sample_id, z = combine_samples(train_samples, sample_ids, train_sample_flags)
		valid_phenotypes = train_phenotypes                        
		
		# split into train-validation partitions
		eval_folds = 5
		kf = StratifiedKFold(sample_id, eval_folds)
		train_indices, valid_indices = next(iter(kf))
		X_train, id_train, z_train = X[train_indices], sample_id[train_indices], z[train_indices]
		X_valid, id_valid , z_valid = X[valid_indices], sample_id[valid_indices], z[valid_indices]
	
	else:
		sample_ids = range(len(train_phenotypes))
		X_train, id_train, z_train = combine_samples(train_samples, sample_ids, train_sample_flags)
		sample_ids = range(len(valid_phenotypes))
		X_valid, id_valid, z_valid = combine_samples(valid_samples, sample_ids, valid_sample_flags)

	if scale:
		z_scaler = StandardScaler()
		z_scaler.fit(X_train)
		X_train = z_scaler.transform(X_train)
	else:
		scaler = None
	
	X_train, z_train, id_train = shuffle(X_train, z_train, id_train)
	train_phenotypes = np.asarray(train_phenotypes)
	y_train = train_phenotypes[id_train]

	if (valid_samples is not None) or generate_valid_set:
		if scale:
			X_valid = z_scaler.transform(X_valid)

		X_valid, z_valid, id_valid = shuffle(X_valid, z_valid, id_valid)
		valid_phenotypes = np.asarray(valid_phenotypes)
		y_valid = valid_phenotypes[id_valid]


	# number of measured markers
	nmark = X_train.shape[1]
   
	# generate multi-cell inputs
	print 'Generating multi-cell inputs...'

	if subset_selection == 'outlier':
		
		# here we assume that class 0 is always the control class
		x_ctrl_train = X_train[y_train == 0]
		to_keep = int(0.1 * (X_train.shape[0] / len(train_phenotypes)))
		nsubset_ctrl = nsubset / np.sum(train_phenotypes == 0)

		# allow each class to have different number of subsets
		nsubset_biased = [0]
		for pheno in range(1, len(np.unique(train_phenotypes))):
			nsubset_biased.append(nsubset / np.sum(train_phenotypes == pheno))
		
		X_tr, y_tr = generate_biased_subsets(X_train, train_phenotypes, id_train, x_ctrl_train,
											nsubset_ctrl, nsubset_biased, ncell, to_keep,
											id_ctrl=np.where(train_phenotypes == 0)[0],
											id_biased=np.where(train_phenotypes != 0)[0])

		# save those because it takes long to generate
		#np.save(os.path.join(outdir, 'X_tr.npy'), X_tr)
		#np.save(os.path.join(outdir, 'y_tr.npy'), y_tr)
		#X_tr = np.load(os.path.join(outdir, 'X_tr.npy'))
		#y_tr = np.load(os.path.join(outdir, 'y_tr.npy'))
		
		if (valid_samples is not None) or generate_valid_set:
			
			x_ctrl_valid = X_valid[y_valid == 0]
			nsubset_ctrl = nsubset / np.sum(valid_phenotypes == 0)
			
			# allow each class to have different number of subsets
			nsubset_biased = [0]
			for pheno in range(1, len(np.unique(valid_phenotypes))):
				nsubset_biased.append(nsubset / np.sum(valid_phenotypes == pheno))

			to_keep = int(0.1 * (X_valid.shape[0] / len(valid_phenotypes)))
			X_v, y_v = generate_biased_subsets(X_valid, valid_phenotypes, id_valid, x_ctrl_valid,
												nsubset_ctrl, nsubset_biased, ncell, to_keep,
												id_ctrl=np.where(valid_phenotypes == 0)[0],
												id_biased=np.where(valid_phenotypes != 0)[0])

			# save those because it takes long to generate
			#np.save(os.path.join(outdir, 'X_v.npy'), X_v)
			#np.save(os.path.join(outdir, 'y_v.npy'), y_v)
			#X_v = np.load(os.path.join(outdir, 'X_v.npy'))
			#y_v = np.load(os.path.join(outdir, 'y_v.npy'))

		else:
			cut = X_tr.shape[0] / 5
			X_v = X_tr[:cut]
			y_v = y_tr[:cut]
			X_tr = X_tr[cut:]
			y_tr = y_tr[cut:]
											
	# TODO: right now equal number of subsets is drawn from each sample
	# Do it per phenotype instead? 
	elif subset_selection == 'kmeans':
		X_tr, y_tr = generate_subsets(X_train, train_phenotypes, id_train,
										nsubset, ncell, k_init=True)
		if (valid_samples is not None) or generate_valid_set:
			X_v, y_v = generate_subsets(X_valid, valid_phenotypes, id_valid,
										nsubset/2, ncell, k_init=True)
	else:
		X_tr, y_tr = generate_subsets(X_train, train_phenotypes, id_train,
										nsubset, ncell, k_init=False)
		if (valid_samples is not None) or generate_valid_set:
			X_v, y_v = generate_subsets(X_valid, valid_phenotypes, id_valid,
										nsubset/2, ncell, k_init=False)

	print 'Done.'
	## neural network configuration ##
	
	# batch size
	bs = 100

	# keras needs (nbatch, ncell, nmark)
	X_tr = np.swapaxes(X_tr, 2, 1)
	X_v = np.swapaxes(X_v, 2, 1)

	if not regression:
		y_tr = to_categorical(y_tr)
		y_v = to_categorical(y_v) 

	# train some neural networks with different parameter configurations
	w_store = dict() 
	accuracies = np.empty(nrun)
	config = dict()
	config['sigma'] = []
	config['nfilter'] = []
	config['learning_rate'] = []
	config['ncell_pooled'] = []

	for irun in range(nrun):
		
		if verbose:
			print 'training network: %d' % (irun + 1)

		if mode == 'supervised':

			sigma = 10 ** np.random.uniform(-2, -1)
			learning_rate = 10 ** np.random.uniform(-3, -2)
			nfilter = np.random.choice(nfilter_choice)
			k = np.random.choice(ncell_pooled)

			config['sigma'].append(sigma)
			config['nfilter'].append(nfilter)
			config['learning_rate'].append(learning_rate)
			config['ncell_pooled'].append(k)
			print 'Cells pooled: %d' % k

			data_input = Input(shape=(ncell, nmark))
			noisy_input = GaussianNoise(sigma=sigma)(data_input)
			conv1 = Convolution1D(nfilter, 1, activation='linear', 
								 W_regularizer=l1l2(l1=coeff_l1, l2=coeff_l2),
								 activity_regularizer=activity_KL(l=coeff_activity, p=0.05),
								 name='conv1')(noisy_input)
			conv1 = Activation('relu')(conv1)

			# the cell grouping part
			pool1 = Lambda(select_top, output_shape=(nfilter,), arguments={'k':k})(conv1)
			if dropout:
				pool1 = Dropout(p=.5)(pool1)
			#pool1 = AveragePooling1D(pool_length=ncell_pooled, stride=ncell_pooled, name='pool1')(select1)
			#pool1 = MaxPooling1D(pool_length=ncell, stride=1, name='pool1')(conv1)
			#pool1 = Flatten()(pool1)

			# network prediction output
			if not regression:
				n_classes = len(np.unique(train_phenotypes))
				output = Dense(n_classes, activation='softmax', W_regularizer=l1l2(l1=coeff_l1, l2=coeff_l2),
								name='output')(pool1)
			else:
				output = Dense(1, activation='tanh', W_regularizer=l1l2(l1=coeff_l1, l2=coeff_l2),
								name='output')(pool1)

			model = Model(input=data_input, output=output)

			if not regression:
				model.compile(optimizer=Adam(lr=learning_rate),
							  loss='categorical_crossentropy',
							  metrics=['accuracy'])
			else:
				model.compile(optimizer=Adam(lr=learning_rate),
							  loss='mean_squared_error')

			filepath = os.path.join(outdir, 'nnet_run_%d.hdf5' % irun)

			check = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, mode='auto')
			earlyStopping = EarlyStopping(monitor='val_loss', patience=patience, mode='auto')
			if not regression:
				model.fit(float32(X_tr), int32(y_tr),
					 		nb_epoch=max_epochs, batch_size=bs, callbacks=[check, earlyStopping],
					 		validation_data=(float32(X_v), int32(y_v)))
			else:
				model.fit(float32(X_tr), float32(y_tr),
					 		nb_epoch=max_epochs, batch_size=bs, callbacks=[check, earlyStopping],
					 		validation_data=(float32(X_v), float32(y_v)))


			# load the model from the epoch with highest validation accuracy
			model.load_weights(filepath)

			if not regression:
				valid_metric = model.evaluate(float32(X_v), int32(y_v))[-1]
				print 'Best validation accuracy: %.2f' % valid_metric
				accuracies[irun] = valid_metric

			else:
				valid_metric = model.evaluate(float32(X_tr), float32(y_tr), batch_size=bs)
				print 'Best train loss: %.2f' % valid_metric
				valid_metric = model.evaluate(float32(X_v), float32(y_v), batch_size=bs)
				print 'Best validation loss: %.2f' % valid_metric
				accuracies[irun] = - valid_metric

			# extract the network parameters
			w_store[irun] = model.get_weights()
			#for iw, w_array in enumerate(model.get_weights()):
			#	print iw, w_array.shape

		# altervatively run in unsupervised mode
		else:
			
			sigma = np.random.uniform(low=0.1, high=0.5)
			nfilter = np.random.choice([10, 20, 50, 100])
			learning_rate = 10 ** np.random.uniform(-3, -1)
			config['sigma'].append(sigma)
			config['nfilter'].append(nfilter)
			config['learning_rate'].append(learning_rate)
		
			data_input = Input(shape=(ncell, nmark))
			noisy_input = GaussianNoise(sigma=sigma)(data_input)
			conv1 = Convolution1D(nfilter, 1, activation='sigmoid', 
			                     W_regularizer=l1l2(l1=coeff_l1, l2=coeff_l2),
			                     name='conv1')(noisy_input)
			
			# the autoencoder part
			upconv1 = Convolution1D(nmark, 1,
						W_regularizer=l1l2(l1=coeff_l1, l2=coeff_l2),
						name='upconv1', activation='linear')(conv1)

			model = Model(input=data_input, output=upconv1)
			model.compile(optimizer=Adam(lr=learning_rate), loss='mean_squared_error')

			filepath = os.path.join(outdir, 'ae_run_%d.hdf5' % irun)
			check = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, mode='auto')
			earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
			model.fit(float32(X_tr), float32(X_tr),
					 nb_epoch=max_epochs, batch_size=bs, callbacks=[check, earlyStopping],
					 validation_data=(float32(X_v), float32(X_v)))

			# load the model from the epoch with highest validation accuracy
			model.load_weights(filepath)
			valid_metric = model.evaluate(float32(X_v), float32(X_v))
			print 'Best validation loss: %.2f' % valid_metric
			accuracies[irun] = - valid_metric

			# extract the network parameters
			w_store[irun] = model.get_weights()
	
	# which filter weights should we return
	# 'best': return the filter weights of the model with highest validation accuracy
	# 'consensus': return consensus filters based on hierarchical clustering
	
	w_best_net, w_cons, cluster_res, sparse_filters = None, None, None, None
	filter_idx = None
	tau = None
	best_net = w_store[np.argmax(accuracies)]
	best_accuracy_idx = np.argmax(accuracies)

	if mode == 'supervised':
		w_best_net = keras_param_vector(best_net, regression)

		if regression and (select_filters == 'best'):
			tau = get_filters_regression(w_best_net, z_scaler, valid_samples, list(valid_phenotypes))
		elif select_filters == 'consensus':
			w_cons, cluster_res = compute_consensus_profiles(w_store, accuracies, accur_thres,
									regression, prioritize=False,
									dendrogram_cutoff=dendrogram_cutoff)
		elif (not regression) and (select_filters == 'best'):
			w_cons, filter_idx = get_filters_classification(w_best_net, z_scaler, valid_samples,
									valid_phenotypes)
		else:
			print 'Returning the only best neural network.'

	results = {
		'clustering_result': cluster_res,
		'best_net': best_net,
		'w_best_net': w_best_net,
		'selected_filters': w_cons,
		'tau': tau,
		'selected_filter_idx' : filter_idx,
		'accuracies': accuracies,
		'best_model_idx': best_accuracy_idx,
		'config': config,
		'labels': labels,
		'scaler': z_scaler
		}
	
	return results


def relu(x):
	return x * (x > 0)


# use the validation samples to select good filters
# for each filter, choose top 30 cells from each class
# to get an estimate of the activation differences between classes

def get_filters_classification(filters, scaler, valid_samples, valid_phenotypes):

	nmark = valid_samples[0].shape[1]
	n_classes = len(np.unique(valid_phenotypes))
	selected_filters = np.zeros((n_classes-1, nmark))
	ntop = 30
	
	valid_ctrl = np.vstack([x for x, y in zip(valid_samples, valid_phenotypes) if y == 0])	
	x0 = scaler.transform(valid_ctrl)
	filter_idx = []

	for i in range(1, n_classes):
		x1 = np.vstack([x for x, y in zip(valid_samples, valid_phenotypes) if y == i])
		x1 = scaler.transform(x1)
		idx = -1
		max_diff = -1

		for ii, foo in enumerate(filters):

			# if this filter is positive for the class of interest
			# and negative for the control class
			if (foo[nmark+1] < 0) and (foo[nmark+1+i] > 0):
				w, b = foo[:nmark], foo[nmark]
				g0 = relu(np.sum(w.reshape(1,-1) * x0, axis=1) + b)
				g1 = relu(np.sum(w.reshape(1,-1) * x1, axis=1) + b)
				d = np.sum(np.sort(g1)[-ntop:]) - np.sum(np.sort(g0)[-ntop:])

				if d > max_diff:
					max_diff = d
					idx = ii

		# now we have the best filter for this specific class
		filter_idx.append(idx)
		selected_filters[i-1] = filters[idx, :nmark]

	return selected_filters, filter_idx


def get_filters_regression(filters, scaler, valid_samples, valid_phenotypes):

	nmark = valid_samples[0].shape[1]
	nsample = len(valid_samples)
	tau = np.zeros((len(filters), 1))
	
	for ii, foo in enumerate(filters):
		w, b, w_out = foo[:nmark], foo[nmark], foo[-1]

		y_pred = np.zeros(nsample)
		for jj, x in enumerate(valid_samples):
			x = scaler.transform(x.copy())
			y_pred[jj] = w_out * np.mean(relu(np.sum(w.reshape(1,-1) * x, axis=1) + b))

		# compute Kendall's tau for filter ii
		tau[ii, 0] = kendalltau(y_pred, np.array(valid_phenotypes))[0]

	return tau

