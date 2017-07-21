# Kaggle Invasive Species Monitoring
This repository is my workflow for Kaggle's Invasive Species Monitoring competition.<br />
In this competition, we are given images with and without *Hydrengea,* a species which is invasive in North America.<p />

## Running the code
### Setting up the environment
First, clone the repository<br />
`git clone https://github.com/tetelestia/kaggle-invasive-species`<br />
Move into the new directory<br />
`cd kaggle-invasive-species`<br />
Create a virtual environment with Python 3 (3.5 or greater)<br />
`virtualenv -p python3 invasive_env`<br />
Then, activate the virtualenv and install all required packages<br />
`source invasive_env/bin/activate`<br />
`pip install -r requirements.txt`<br />
Finally, PyTorch must be installed from the website. Go to pytorch.org and follow the installation instructions on their website.<br />
If you are running on an Nvidia GPU (highly suggested), CUDA must be installed in order to use the GPU, and while you're at it, installing cuDNN is a good idea, as it decreases computation time by about a factor of two.

### Directory structure
The data is to be set in folders as such:<br />
 - data
   - train/*
   - test/*
   - sample_submission.csv
   - train_labels.csv
