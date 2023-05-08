
# Computer Vision - MAIBI

This repository contains the notebooks for the course of Computer Vision, given 
at the master AI for Business and Industry.

# Setting up VSC OnDemand

- Browse to <https://ondemand.hpc.kuleuven.be/> and log in
- Click on *Login Server Shell Access*
- Test if Conda is installed by running `conda --version`. If this runs without an error message containing something like `command not found`, Conda is installed.
- If you do get a `command not found` error, you can install Conda as follows:

```bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $VSC_DATA/miniconda3
echo 'export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"' >> ~/.bashrc
source ~/.bashrc
```

- Run the following commands:

```bash
cd $VSC_DATA
git clone https://github.com/florisdf/maibi_ba.git
cd maibi_ba
conda create -n maibi_ba r-essentials r-base
source activate maibi_ba
Rscript install.R
```

- Go back to <https://ondemand.hpc.kuleuven.be/>
- Click on *Jupyter Lab*
- Fill in the fields:
    - Partition: *interactive*
    - Number of hours: *4* (can have a value up to *16*)
    - Number of nodes: *1*
    - Required memory per core in megabytes: *3400*
    - Number of cores: 4
    - Number of GPUs: 0
- Click *Launch*
- Once Jupyter is running, click *Connect to Jupyter Lab*

- For the notebooks in the directory `sentiment`, you need to choose the *R* kernel `maibi_ba`.
- For the notebooks in the directory `cost_sensitive_learning`, you need to choose the *Python* kernel `Python 3 (ipykernel)`.
