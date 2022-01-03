# EvoLSTM
Sequence to Sequence LSTM based evolution simulator
EvoLSTM requires external Nvidia GPU to train and simulate sequence evolution.

# Cloning the repository
$ git clone https://github.com/DongjoonLim/EvoLSTM.git

# Make required directories.
$ mkdir data
$ mkdir prepData
$ mkdir models
$ mkdir simulation

# Downloading training data.
The sequence alginment maf file are available at http://repo.cs.mcgill.ca/PUB/blanchem/Boreoeutherian/ download the maf files and put it under the 'data' directory.
The tree structure and nomenclature for species are available at http://hgdownload.cse.ucsc.edu/goldenpath/hg38/multiz100way/hg38.100way.nh
The ancestral sequences are labelled as _(First character seq1)(First character seq2), so the most recent common ancestor of hp38 and pantro4 will be _HP.

# Install requirements
$ pip install -r requirements.txt

# Preprocessing sequences to have meta-nucleotides.
Run the python file with the following command \
$ python3 prep_insert2.py chromosome ancName desName \
So if you want to preprocess the human chromosome 2 portion of the most recent common ancestor of hp38 and pantro4 evolving to hg38, the command will be \
$ python3 prep_insert2.py 2 _HP hg38 \

# Training EvoLSTM with preprocessed sequences.
Run the python file with \
$ python3 insert2Train_general.py ancName desName train_size seq_length \
where the train_size is the length of the training sequence (Due to the memory constraints), trimming out sequences longer than this number, recommended to start from 100000 \
the seq_length is the length of the sequence context. Recommended to set this to 15 \

so an example would be \
$ python3 insert2Train_general.py 2 _HP hg38 100000, 15 \

# Simulating sequence evolution 
Run the python file with \
$ python3 simulate.py ancName desName sample_size gpu_index chromosome  \
where the sample_size is the length of the desired input sequence length \
gpu_index is the index of the gpu card you want to run the simulation on. Can be found with the command nvidia-smi. If there is only one gpu, set it to 0 \
chromosome is the desired chromosome the simulation will be run on. \
So an example would be, \
$ python3 simulate.py $_HP$ hg38 100000 0 2 \
Which will simulate the first 100000 bps of _HP sequence in chromosome 2. \
The output of the simulation will be saved as 'simulated_{}_{}_{}.npy' \
Use numpy to load and read this npy files


