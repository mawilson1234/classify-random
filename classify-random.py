# svm.py
#
# fit a svm to see if a hyperplane exists between randomly generated token embeddings
import hydra
import numpy as np
import logging

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from math import floor
from typing import *
from omegaconf import DictConfig

from scipy.stats import sem

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

log = logging.getLogger(__name__)

def create_randomly_grouped_embeddings(ngroups: int, nembeddings: int, ndims: int) -> Tuple[np.ndarray, List[int]]:
	'''
	Creates equal-size groups of random embeddings
	
		params:
			ngroups (int)			: the number of groups to assign embeddings to
			nembeddings (int)		: the number of random embeddings ot generate
			ndims (int)				: the length of the random embeddings
		
		returns:
			embeddings (np.ndarray) : randomly generated embeddings of the shape (nembeddings, ndims)
			labels (List[int])		: integer labels assigned to the random embeddings
	'''
	if not nembeddings % ngroups == 0:
		log.warning(f'{nembeddings} cannot be split into {ngroups} equal-size groups.')
		return
	
	embeddings 	= np.random.rand(nembeddings,ndims)
	labels 		= [i for i in range(ngroups) for _ in range(int(nembeddings/ngroups))]
	
	return embeddings, labels

def run_trials(ntrials: int, ngroups: int, ntrain: int, ntest: int, ndims: int) -> None:
	'''
	Generates random embeddings and fits a classifier. Logs average accuracy.
	
		params:
			ntrials (int)	: the number of trials to run
			ngroups (int)	: the number of groups to assign embeddings to
			ntrain (int)	: the number of random embeddings to generate per training set
			ntest (int) 	: the number of random embeddings to generate per test set
			ndims (int)		: the length of the random embeddings
	'''
	log_filename = [handler for handler in log.root.handlers if isinstance(handler, logging.FileHandler)][0]
	log_filename = log_filename.stream.name.replace(hydra.utils.get_original_cwd(), '')
	log.info(f'Saving to "{log_filename}"')
	log.info(f'Running {ntrials} trials with ngroups={ngroups}, ntrain={ntrain}, ntest={ntest}, ndims={ndims}')
	
	train_accuracies 	= []
	test_accuracies 	= []
	with logging_redirect_tqdm():
		try:
			for i in tqdm(range(ntrials)):
				train_inputs, train_labels 	= create_randomly_grouped_embeddings(ngroups, ntrain, ndims)
				classifier 					= svm.SVC(kernel='linear')
				classifier.fit(train_inputs, train_labels)
				
				train_accuracy				 = get_accuracy(classifier, train_inputs, train_labels)
				train_accuracies.append(train_accuracy)
				
				test_inputs, test_labels 	= create_randomly_grouped_embeddings(ngroups, ntest, ndims)
				test_accuracy 				= get_accuracy(classifier, test_inputs, test_labels)
				test_accuracies.append(test_accuracy)
				
				mean_train_accuracy 		= np.mean(train_accuracy)
				mean_test_accuracy 			= np.mean(test_accuracies)
				
				if i > 0:
					sem_train_accuracy 		= sem(train_accuracies)
					sem_test_accuracy 		= sem(test_accuracies)
				else:
					sem_train_accuracy 		= np.nan
					sem_test_accuracy 		= np.nan
				
				if i % 50 == 49:
					log.info(f'Mean train, test accuracy at trial {str(len(train_accuracies)).rjust(len(str(ntrials)))}. Train: {mean_train_accuracy:.2f} (\u00b1{sem_train_accuracy:.2f}), test: {mean_test_accuracy:.2f} (\u00b1{sem_test_accuracy:.2f})')
				
		except KeyboardInterrupt:
			log.warning(f'Halted manually at trial {str(len(train_accuracies)).rjust(len(str(ntrials)))}')
			mean_train_accuracy 			= np.mean(train_accuracies)
			sem_train_accuracy 				= sem(train_accuracies)
			
			# account for cases when we've finished training but not testing at an iteration
			mean_test_accuracy 				= np.mean(test_accuracies[:len(train_accuracies)])
			sem_test_accuracy 				= sem(test_accuracies[:len(train_accuracies)])
			
			log.info(f'Mean train, test accuracy at trial {str(len(train_accuracies)).rjust(len(str(ntrials)))}. Train: {mean_train_accuracy:.2f} (\u00b1{sem_train_accuracy:.2f}), test: {mean_test_accuracy:.2f} (\u00b1{sem_test_accuracy:.2f})')

def get_accuracy(classifier: svm.SVC, inputs: np.ndarray, labels: List[int]) -> float:
	'''
	Gets accuracy score for classifier predictions.
	
		params:
			classifier (svm.SVC)	: a classifier fit to the inputs
			inputs (np.ndarray)		: a set of inputs to check the classifier on
			labels (List[int])		: a list of labels for each input
		
		returns:
			accuracy (float)		: the accuracy with which the model predicts labels from inputs, as a percent
	'''
	predictions = classifier.predict(inputs)
	accuracy 	= accuracy_score(labels, predictions) * 100
	
	return accuracy

@hydra.main(config_path='.', config_name='classify-random')
def main(cfg: DictConfig) -> None:
	'''Runs trials according to the config file/command line config'''
	run_trials(cfg.ntrials, cfg.ngroups, cfg.ntrain, cfg.ntest, cfg.ndims)


if __name__ == '__main__':
	
	main()