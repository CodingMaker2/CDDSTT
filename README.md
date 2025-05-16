Conditional Diffusion Transformer Enables Few-Shot Spatiotemporal Modeling

Our method is implemented based on the BasicTS framework, for more information https://github.com/GestaltCogTeam/BasicTS/

Python
Python 3.6 or higher is required (3.8 or higher is recommended).

We recommend using Miniconda or Anaconda to create a virtual Python environment.

PyTorch
BasicTS is flexible regarding the PyTorch version. You can install PyTorch according to your Python version. We recommend using pip for installation.

Other Dependencies
After ensuring PyTorch is installed correctly, you can install the other dependencies:

pip install -r requirements.txt

Example 1: Python 3.11 + PyTorch 2.3.1 + CUDA 12.1 (Recommended)
## Install Python
conda create -n BasicTS python=3.11
conda activate BasicTS
## Install PyTorch
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
## Install other dependencies
pip install -r requirements.txt

how to run the code:
python CDT/experiments/train.py

