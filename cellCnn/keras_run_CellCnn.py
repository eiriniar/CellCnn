import sys, os
import copy
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from cellCnn.utils import combine_samples, landmark_normalization 
from cellCnn.utils import normalize_outliers_to_control, normalize_outliers
from cellCnn.utils import cluster_profiles, keras_param_vector
from cellCnn.utils import generate_subsets, generate_biased_subsets
from cellCnn.utils import get_filters_classification, get_filters_regression
from cellCnn.theano_utils import select_top, float32, int32, activity_KL

from keras.layers import Input, Dense, Flatten, Reshape, Lambda, Activation, Dropout
from keras.layers.convolutional import Convolution1D, MaxPooling1D, AveragePooling1D
from keras.layers.noise import GaussianNoise
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l1l2, activity_l1
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical


def train_model(train_samples, train_phenotypes, labels, outdir,
				valid_samples=None, valid_phenotypes=None, generate_valid_set=True,
				landmark_norm=None, scale=True, nrun=10, regression=False,
				ncell=500, nsubset=4000, per_sample=False, subset_selection='random', 
				ncell_pooled=[1,3,5],  nfilter_choice=[2,3,4,5],
				learning_rate=None, coeff_l1=0, coeff_l2=1e-4, dropout='auto', dropout_p=.5,
				noisy_input=False, coeff_activity=0, max_epochs=50, patience=10, verbose=1,
				select_filters='consensus', dendrogram_cutoff=0.4, accur_thres=.95):
	
	"""Performs a CellCnn analysis.

	# Arguments

	train_samples : list with input samples (e.g. cytometry samples) as numpy arrays
	train_phenotypes : list of phenotypes associated with the samples in train_samples
	labels : names of measured markers in train_samples
	outdir : directory where output will be generated
	
	valid_samples : list with samples to be used as validation set while training the network
	valid_phenotypes : list of phenotypes associated with the samples in valid_samples
	generate_valid_set : if valid_samples is not provided, generate a validation set from the train_samples
	scale : whether to z-transform each feature (mean=0, std=1) prior to training

	nrun  : number of neural network configurations to try
	regression : set to True for a regression problem. Default is False, which corresponds to a classification
				 setting.

	ncell : number of cells per multi-cell input
	nsubset : total number of multi-cell inputs that will be generated per class, 
			  if per_sample = False
			  total number of multi-cell inputs that will be generated from each input sample,
			  if per_sample = True
	per_sample : whether the nsubset argument refers to each class or each input sample
	             for regression problems, it is automatically set to True
	subset_selection : can be 'random' or 'outlier'. Generate multi-cell inputs uniformly at random
					   or biased towards outliers. The latter option is only relevant for detection of
					   extremely rare (frequency < 0.1%) cell populations.

	ncell_pooled : A list specifying candidate numbers of cells that will be max-pooled per filter.
				   For mean pooling, set ncell_pooled = [ncell].	
	nfilter_choice : a list specifying candidate numbers of filters for the neural network

	learning_rate : learning rate for the Adam optimization algorithm. 
				    If None learning rates in the range [0.001, 0.01] will be tried out.
	dropout : whether to use dropout (at each epoch, set a neuron to zero with probability dropout_p)
	          The default behavior 'auto' uses dropout when nfilter > 5.
	dropout_p : dropout probability
	coef_l1 : coefficiant for L1 weight regularization
	coef_l2 : coefficiant for L2 weight regularization
	coeff_activity : coefficiant for regularizing the activity at each filter
	noisy_input : if True, a Gaussian noise layer is added after the input layer

	max_epochs : maximum number of iterations through the data
	patience : number of epochs before early stopping 
			  (stops if the validation loss does not decrease anymore)

	dendrogram_cutoff : cutoff for hierarchical clustering of filter weights. Clustering is performed
	 					using cosine similarity, so the cutof should be in [0, 1]. A lower cutoff 
	 					will generate more clusters. 
						  
	accur_thres : keep filters from models achieving at least this accuracy. If less than 3 models
	 			  pass the accuracy threshold, keep filters from the best 3 models.

	# Returns

	results : a dictionary with the following entries
		
		'clustering_result': clustered filter weights from all runs achieving 
							validation accuracy above the specified threshold 'accur_thres'
		'selected_filters': a consensus filter metrix from the above clustering result
		'best_net': the best model (achieving highest validation accuracy)
		'w_best_net': filter and output weights of the best model
		'accuracies': list of validation accuracies achieved by different models
		'best_model_index': list index of the best model
		'config': list of neural network configurations used
		'labels': names of measured markers
		'scaler': a z-transform scaler object fitted to the training data

	"""
	
	# copy the list of samples so that they are not modified in place
	train_samples = copy.deepcopy(train_samples)
	if valid_samples is not None:
		valid_samples = copy.deepcopy(valid_samples)

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

	# merge all input samples (X_train, X_valid)
	# and generate an identifier for each of them (train_id, valid_id)
	if (valid_samples is None) and (not generate_valid_set):
		sample_ids = range(len(train_phenotypes))
		X_train, id_train = combine_samples(train_samples, sample_ids)
		
	elif (valid_samples is None) and generate_valid_set:
		sample_ids = range(len(train_phenotypes))
		X, sample_id = combine_samples(train_samples, sample_ids)
		valid_phenotypes = train_phenotypes                        
		
		# split into train-validation partitions
		eval_folds = 5
		kf = StratifiedKFold(sample_id, eval_folds)
		train_indices, valid_indices = next(iter(kf))
		X_train, id_train = X[train_indices], sample_id[train_indices]
		X_valid, id_valid = X[valid_indices], sample_id[valid_indices]
	
	else:
		sample_ids = range(len(train_phenotypes))
		X_train, id_train = combine_samples(train_samples, sample_ids)
		sample_ids = range(len(valid_phenotypes))
		X_valid, id_valid = combine_samples(valid_samples, sample_ids)

	if scale:
		z_scaler = StandardScaler()
		z_scaler.fit(X_train)
		X_train = z_scaler.transform(X_train)
	else:
		scaler = None
	
	X_train, id_train = shuffle(X_train, id_train)
	train_phenotypes = np.asarray(train_phenotypes)

	# an array containing the phenotype for each single cell
	y_train = train_phenotypes[id_train]

	if (valid_samples is not None) or generate_valid_set:
		if scale:
			X_valid = z_scaler.transform(X_valid)

		X_valid, id_valid = shuffle(X_valid, id_valid)
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

		# generate a fixed number of subsets per class
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
			
			# generate a fixed number of subsets per class
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
											
	else:

		# generate 'nsubset' multi-cell inputs per input sample
		if per_sample:
			X_tr, y_tr = generate_subsets(X_train, train_phenotypes, id_train,
											nsubset, ncell)
			if (valid_samples is not None) or generate_valid_set:
				X_v, y_v = generate_subsets(X_valid, valid_phenotypes, id_valid,
											nsubset, ncell)

		# generate 'nsubset' multi-cell inputs per class
		else:
			nsubset_list = []
			for pheno in range(len(np.unique(train_phenotypes))):
				nsubset_list.append(nsubset / np.sum(train_phenotypes == pheno))
			X_tr, y_tr = generate_subsets(X_train, train_phenotypes, id_train,
										  nsubset_list, ncell)

			if (valid_samples is not None) or generate_valid_set:
				nsubset_list = []
				for pheno in range(len(np.unique(valid_phenotypes))):
					nsubset_list.append(nsubset / np.sum(valid_phenotypes == pheno))
				X_v, y_v = generate_subsets(X_valid, valid_phenotypes, id_valid,
											nsubset_list, ncell)
	print 'Done.'
	
	## neural network configuration ##
	
	# batch size
	bs = 100

	# keras needs (nbatch, ncell, nmark)
	X_tr = np.swapaxes(X_tr, 2, 1)
	X_v = np.swapaxes(X_v, 2, 1)
	n_classes = 1

	if not regression:
		n_classes = len(np.unique(train_phenotypes))
		if n_classes > 2:
			y_tr = to_categorical(y_tr)
			y_v = to_categorical(y_v) 

	# train some neural networks with different parameter configurations 
	accuracies = np.empty(nrun)
	w_store = dict()
	config = dict()
	config['sigma'] = []
	config['nfilter'] = []
	config['learning_rate'] = []
	config['ncell_pooled'] = []
	lr = learning_rate

	for irun in range(nrun):
		
		if verbose:
			print 'training network: %d' % (irun + 1)

		if noisy_input:
			sigma = 10 ** np.random.uniform(-2, -1)
			config['sigma'].append(sigma)
		if learning_rate is None:
			lr = 10 ** np.random.uniform(-3, -2)
			config['learning_rate'].append(lr)

		# choose number of filters for this run
		nfilter = np.random.choice(nfilter_choice)
		config['nfilter'].append(nfilter)
		print 'Number of filters: %d' % nfilter

		# choose number of cells pooled for this run
		k = np.random.choice(ncell_pooled)
		config['ncell_pooled'].append(k)
		print 'Cells pooled: %d' % k

		# the input layer
		data_input = Input(shape=(ncell, nmark))
		if noisy_input:
			data_input = GaussianNoise(sigma=sigma)(data_input)
			
		# the filters
		conv1 = Convolution1D(nfilter, 1, activation='linear', 
							W_regularizer=l1l2(l1=coeff_l1, l2=coeff_l2),
							activity_regularizer=activity_KL(l=coeff_activity, p=0.05),
							name='conv1')(data_input)
		conv1 = Activation('relu')(conv1)

		# the cell grouping part
		pool1 = Lambda(select_top, output_shape=(nfilter,), arguments={'k':k})(conv1)
		#pool1 = AveragePooling1D(pool_length=ncell_pooled, stride=ncell_pooled, name='pool1')(select1)
		#pool1 = MaxPooling1D(pool_length=ncell, stride=1, name='pool1')(conv1)
		#pool1 = Flatten()(pool1)

		# possibly add dropout
		if (dropout == True) or ((dropout == 'auto') and (nfilter > 5)):
			pool1 = Dropout(p=dropout_p)(pool1)

		# network prediction output
		if not regression:
			if n_classes == 2:
				output = Dense(1, activation='sigmoid', W_regularizer=l1l2(l1=coeff_l1, l2=coeff_l2),
								name='output')(pool1)
			else:
				output = Dense(n_classes, activation='softmax', W_regularizer=l1l2(l1=coeff_l1, l2=coeff_l2),
								name='output')(pool1)
		else:
			output = Dense(1, activation='tanh', W_regularizer=l1l2(l1=coeff_l1, l2=coeff_l2),
							name='output')(pool1)

		model = Model(input=data_input, output=output)

		if not regression:
			if n_classes == 2:
				model.compile(optimizer=Adam(lr=lr),
							  loss='binary_crossentropy',
							  metrics=['accuracy'])
			else:
				model.compile(optimizer=Adam(lr=lr),
							  loss='categorical_crossentropy',
							  metrics=['accuracy'])
		else:
			model.compile(optimizer=Adam(lr=lr),
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
			train_metric = model.evaluate(float32(X_tr), float32(y_tr), batch_size=bs)
			print 'Best train loss: %.2f' % train_metric
			valid_metric = model.evaluate(float32(X_v), float32(y_v), batch_size=bs)
			print 'Best validation loss: %.2f' % valid_metric
			accuracies[irun] = - valid_metric

		# extract the network parameters
		w_store[irun] = model.get_weights()

	# the best performing network and the accuracy it achieves
	best_net = w_store[np.argmax(accuracies)]
	best_accuracy_idx = np.argmax(accuracies)

	# weights from the best-performing network
	w_best_net = keras_param_vector(best_net, regression)

	# post-process the learned filters	
	# cluster weights from all networks that achieved accuracy above the specified thershold 
	w_cons, cluster_res = cluster_profiles(w_store, accuracies, accur_thres,
											regression, dendrogram_cutoff=dendrogram_cutoff)

	results = {
		'clustering_result': cluster_res,
		'selected_filters': w_cons,
		'best_net': best_net,
		'w_best_net': w_best_net,
		'accuracies': accuracies,
		'best_model_index': best_accuracy_idx,
		'config': config,
		'labels': labels,
		'scaler': z_scaler
	}	

	if valid_samples is not None:
		if regression:
			tau = get_filters_regression(w_best_net, z_scaler, valid_samples, list(valid_phenotypes))
			results['filter_tau'] = tau

		else:
			filter_w, filter_idx = get_filters_classification(w_best_net, z_scaler, valid_samples,
															  valid_phenotypes)
			results['selected_filters_supervised'] = filter_w
			results['selected_filters_supervised_indices'] = filter_idx
	
	return results
