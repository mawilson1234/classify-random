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

def create_randomly_grouped_embeddings(ngroups: int = None, nembeddings: int = None, ndims: int = None) -> Tuple[np.ndarray, List[int]]:
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

def run_trials(ntrials: int = None, ngroups: int = None, nembeddings: int = None, ndims: int = None) -> None:
	'''
	Generates random embeddings and fits a classifier. Logs average accuracy.
	
		params:
			ntrials (int)		: the number of trials to run
			ngroups (int)			: the number of groups to assign embeddings to
			nembeddings (int)		: the number of random embeddings ot generate
			ndims (int)				: the length of the random embeddings
	'''
	
	log.info(f'Running {ntrials} trials with ngroups={ngroups}, nembeddings={nembeddings}, ndims={ndims}')
	
	accuracies = []
	with logging_redirect_tqdm():
		try:
			for i in tqdm(range(ntrials)):
				inputs, labels 	= create_randomly_grouped_embeddings(ngroups, nembeddings, ndims)
				classifier 		= svm.SVC(kernel='linear')
				classifier.fit(inputs, labels)
				
				accuracy = get_accuracy(classifier, inputs, labels)
				accuracies.append(accuracy)
				mean_accuracy = np.mean(accuracies)
				if i > 0:
					sem_accuracy  = sem(accuracies)
				else:
					sem_accuracy = np.nan
				
				if i % 50 == 49:
					log.info(f'Mean accuracy at trial {str(len(accuracies)).rjust(len(str(ntrials)))}: {mean_accuracy:.2f} (\u00b1{sem_accuracy:.2f})')
				
		except KeyboardInterrupt:
			log.warning(f'Halted manually at trial {str(len(accuracies)).rjust(len(str(ntrials)))}')
			avg_accuracy = np.mean(accuracies)
			sem_accuracy = sem(accuracies)
			log.info(f'Mean accuracy at trial {str(len(accuracies)).rjust(len(str(ntrials)))}: {avg_accuracy:.2f} (\u00b1{sem_accuracy:.2f})')

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
	run_trials(cfg.ntrials, cfg.ngroups, cfg.nembeddings, cfg.ndims)


if __name__ == '__main__':
	
	main()