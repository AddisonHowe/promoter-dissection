# Dissecting the Promoter

## Installation and Setup

Clone the project repository:
```bash
git clone https://github.com/AddisonHowe/promoter-dissection.git && cd promoter-dissection
```

Create a conda environment from the `environment.yml` file.
By default, installation includes optional dependencies for enabling JAX acceleration, jupyter notebooks, and test development.
These can be excluded by editing the `environment.yml` file to remove the optional pip dependencies.

```bash
conda env create -n <env-name> -f environment.yml
conda activate <env-name>
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
