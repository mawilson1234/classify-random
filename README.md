# Classify Random

Examines the performance of linear support vector machines (SVMs) on classifying randomly generated and assigned data.

## Installation

Clone this git repo and run the following to install the required (non-standard) dependencies:

  - `pip install hydra-core --upgrade`  
  - `pip3 install numpy tqdm omegaconf scipy sklearn`
  
You can also use `conda env create -f environment.yaml` and `conda activate cls-ran`.

## Usage

From the terminal run:
  - `python3 classify-random.py ntrials=1000 ngroups=2 ntrain=1000 ntest=100 ndims=10`
  
This will run 1000 trials where a linear SVM is fit on 1000 randomly initialized 10-dimensional vectors which are randomly assigned to two equal-size groups. The SVM will do its best to classify the random embeddings according to the random labels for the training data. Then, 100 randomly initialized 10-dimensional vectors will be generated as a test set. Mean accuracy and standard error are logged every 50 iterations for train and test performance separately. Logs will be printed to the console and saved in a file in `logs/` with the name `cls_ran-n1000-g2-train1000-test100-d10-{current_datetime}.log`

You can adjust the settings above by filling in different values for the numbers after the `=`. Here's an explanation of what each option does.

  - `ntrials`: The number of SVMs to fit.
  - `ngroups`: The number of groups to randomly assign equal numbers of generated embeddings to.
  - `ntrain`: The number of embeddings in each training set. Must be a multiple of `ngroups` (because equal numbers of embeddings are assigned to each group).
  - `ntest`: The number of embeddings in each test set. Must be a multiple of `ngroups` (because equal numbers of embeddings are assigned to each group).
  - `ndims`: The dimensionality of the embeddings.
  
You can halt training early by pressing `Ctrl/Cmd+C`. The mean accuracy and standard error as of the final trial before interruption will be logged if you do this.
  
## Explanation

You will notice that if the number of dimensions is a high enough fraction of the number of examples, above chance (i.e., $>\dfrac{ngroups}{100}$) performance on the training set is achieved consistently. Test performance tends to remain at chance (as expected). One should use caution in interpreting the results of fitting an SVM to a few examples of high dimensional data and interpreting the results on the training data. Here, the vectors and groups are randomly generated. A vector encodes no information about its group. Nevertheless, an SVM can be good at distinguishing different random groups if the dimensionality of the embeddings is high enough. In other words, SVMs are good at finding spurious patterns in random data, provided the dimensionality of the data is a high enough fraction of the number of examples.