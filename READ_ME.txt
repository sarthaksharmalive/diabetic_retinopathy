# Diabetic Retinopathy using Convolution Neural Networks

# Sarthak Sharma - 163079010
# Ravisankar Reddy - 163079024

---------------------------------------------------------

# All code have been written by both of us together at
	simultaneous sitting. 

# For Prediction, download the following files and folders:
	1. prediction.py
	2. bicross_resnet_model.hdf5.partaa and bicross_resnet_model.hdf5.partab (two part file)
	3. test/

# Now join the 2 part model file first:
# Open terminal, type: cat bicross_resnet_model.hdf5.parta* >bicross_resnet_model.hdf5

# In terminal, type: python3 prediction.py
	Output: Puts two plots: Confusion Matrix and ROC curve

# All other codes for training, image pre processing and other
	bash scripts, and how to use them are described in 
	themselves only.

# For training, similar directory structure as test/ has to be
	maintained for train/ and validation/

# Steps to follow while training is:
	1. Download dataset (from kaggle) and extract in a folder.
	2. Crop Images using scripts available in repo.
	3. Apply Histogram Equalization using scripts available in repo
	4. Split train, validation samples (check their lables first using
		checkLables.sh) and using train_test_split_resized.sh, split 
		them in their respective directories.
	5. run dr_<training_model>.py to train.