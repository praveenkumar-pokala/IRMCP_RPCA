# IR MCP Minimization for RPCA
This repository is official implementation of `Iteratively Reweighted Minimax Concave Penality Minimization for an Accurate Low-Rank plus Sparse Decomposition`. This repository contains codes for synthetic data and a few real world datasets. Complete codes with datasets and video results are available [here](https://drive.google.com/drive/folders/1_xACjQo1HA5s3px_pUM613B5KUkFV2S3?usp=sharing).


## Instructions to run experiments
Required Python version: 3.6

Install the required packages by running

    pip install -r requirements.txt
### Synthetic Data
    python <method_name>.py
### I2R Dataset
* Download the dataset video into the respective folder
* Create the preprocessed data files by running the below command (.npy files are created)

        python create_data_matrix.py
* Run the python file related to required method

        python <method_name>.py
### CDNet Dataset
* Copy the input and ground truthfolders of canoe to CDNet/canoe directory
* Create the preprocessed data files by running the below command (.npy files are created)

        python create_data_matrix.py
* Run the python file related to required method

        python <method_name>.py
### BMC Dataset
* Copy the 322 video sequence and it's groundtruth to BMC/322 directory
* Create the preprocessed data files by running the below commands (.npy files are created)

        python create_data_matrix.py
        python vid2fr.py
        python create_gt_matrix.py
* Run the python file related to required method

        python <method_name>.py
        

*Any suggestions or issues are openly accepted.*