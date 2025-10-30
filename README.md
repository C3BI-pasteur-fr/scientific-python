# Scientific Python Course (2025)

1. Jupyter
2. Numpy
3. Pandas
5. Seaborn
6. Scipy
7. Statsmodels
8. Scikit-learn

## Materials requirements

We will use jupyter notebook and some python libraries.
The are 2 ways to create a suitable environment for this course:

- from a python environment
- from a conda/mamba environment

But whatever your choice you need to have git installed to get the course materials


### from virtual environment

To create a python environment (to do only once)

	python3 -m venv Scientific_Python
	cd Scientific_Python
	source bin/activate
	git clone https://github.com/C3BI-pasteur-fr/scientific-python.git  Scientific_Python_Course
	cd Scientific_Python_Course
	pip install -r requirements.txt

To exit from the environment

	deactivate

To reactivate the environment

	cd Scientific_Python
	source bin/activate

	# to run jupyter notenook server
	jupyter-lab

Windows users must run `Scripts\activate.bat` (in cmd.exe) or `Scripts\activate.ps1` (in PowerShell) instead of `source bin/activate`.

### from conda/mamba (recommended for windows guys)


To create a Conda/mamba environement you have to install Conda(anaconda or miniconda) or mamba
Then create your environement (to do only once)

	mamba env create -n Scientific_Python -c conda-forge bioconda
	conda activate Scientific_Python

	# get the course materials
	git clone https://github.com/C3BI-pasteur-fr/scientific-python.git Scientific_Python_Course
    cd Scientific_Python_Course

	# install prerequisites
    mamba install --file requirements.txt

To exit from the conda environment/mamba

	conda deactivate

To reactivate the environment

	conda activate Scientific_Python

	# to run jupyter notebook server
	jupyter-lab

