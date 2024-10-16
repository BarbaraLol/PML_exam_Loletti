# PML_exam_Loletti
- Bayesian_dataset_preprocessing.py: is the simplest version run on Orfeo and used to produce the dataset used by the BNN
- Bayesian_dataset_preprocessing_parallelized.py: Parallelized version of the Bayesian_dataset_preprocessing.py in which the number of processes can be set based on the particular features of the machine that will excute the code. Moreover the preprocessing works by deviding the whole starting dataset into small batches which size can be set manually. When the image of the spectrogram is saved, the GPU is involved in order to reduce the workload form the CPU and prevent any kind of possible bottleneck. A simple system is also implemented in order to save intermediate results so to have restart checkpoints
- 
