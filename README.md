# use_roost
A repo that with some changes to Roost for easier scripting.

Please see https://github.com/CompRhys/roost for the original code.

This repo adds the use_roost.py and use_train.py as well as minor changes to data.py.






requirements:

conda install tensorboard
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
where ${CUDA} should be replaced by either cpu, cu92, cu100 or cu101 depending on your PyTorch installation.
