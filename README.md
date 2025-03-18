# Dissecting the Promoter

## Installation and Setup

Clone the project repository:
```bash
git clone https://github.com/AddisonHowe/promoter-dissection.git && cd promoter-dissection
```

Create a conda environment from the `environment.yml` file, and install all required dependencies.

```bash
conda env create -n <env-name> -f environment.yml
conda activate <env-name>
```

By default, installation excludes optional dependencies for enabling JAX acceleration, jupyter notebooks, and test development.
These can be included by editing the `environment.yml` file to add the optional pip dependencies.
Simply comment and uncomment the desired lines nested under `pip`.
Alternatively, after activating the environment, rerun the pip installation with the desired options, for example:

```bash
conda activate <env-name>
# Install optional dependencies and specify editable install.
python -m pip install -e ".[jax,jupyter,dev]" 
```

Assuming all optional dependencies have been installed, check that all tests pass via:

```bash
conda activate <env-name>
pytest tests
```


## Examples


## Acknowledgments

This project was inspired by the work of the [Rob Phillips Lab](https://github.com/RPGroup-PBoC). In particular, the [RegSeq project](http://rpdata.caltech.edu/publications/Ireland2020.pdf).


## References
[1] William T Ireland, Suzannah M Beeler, Emanuel Flores-Bautista, Nicholas S McCarty, Tom Röschinger, Nathan M Belliveau, Michael J Sweredoski, Annie Moradian, Justin B Kinney, Rob Phillips (2020) Deciphering the regulatory genome of Escherichia coli, one hundred promoters at a time eLife 9:e55308.
