# PML_exam_Loletti
- Bayesian_dataset_preprocessing.py: is the simplest version run on Orfeo and used to produce the dataset used by the BNN
- Bayesian_dataset_preprocessing_parallelized.py: Parallelized version of the Bayesian_dataset_preprocessing.py in which the number of processes can be set based on the particular features of the machine that will excute the code. Moreover the preprocessing works by deviding the whole starting dataset into small batches which size can be set manually. When the image of the spectrogram is saved, the GPU is involved in order to reduce the workload form the CPU and prevent any kind of possible bottleneck. A simple system is also implemented in order to save intermediate results so to have restart checkpoints







Implementing history - evolution of the code
1. Preprocessing
2. Parallelized preprocessing
3. BNN - training part
4. Splitting the BNN code for better maintenance
5. Problem resolutions before first real sbatch on Orfeo (dimention.py)
6. Results for Adam({"lr": 0.01}) quite poor so tried to change it to Adam({"lr": 0.001})
7. Results still quite poor so started again from scratch the training using Adam({"lr": 0.0001}) but the accuracy is poor and reducing at some point 
8. Trying with a 3 layers BNN, with 512 neurons, Adam({"lr": 0.0001}) 

Possible features to implement/add to the code
- Data augmentation 
- Eaerly stopping and learning rate schedulers
- Adding gradient clipping



Errors enocuontered
1. 
RuntimeError: mat1 and mat2 shapes cannot be multiplied (128x961450 and 512x512)
           Trace Shapes:                   
            Param Sites:                   
     module$$$out.weight   3 512           
       module$$$out.bias       3           
           Sample Sites:                   
module$$$fc1.weight dist       | 512 961450
                   value       | 512 961450
  module$$$fc1.bias dist       | 512       
                   value       | 512       
module$$$fc2.weight dist       | 512    512
                   value       | 512    512
  module$$$fc2.bias dist       | 512       
                   value       | 512       
module$$$fc3.weight dist       | 512    512
                   value       | 512    512
  module$$$fc3.bias dist       | 512       
                   value       | 512       
               data dist       |           
                   value 128   | 

2. 