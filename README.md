# Final Project Computational Methods - ICTP-SAIFR/Serrapilheira

# Analysis of the impact of vaccination on the dynamics of COVID-19 in Colombia.

In this project you can find an analysis of the dynamics of the disease from mathematical modeling in Python.

## Project structure

```
project/
*    ├── data/
*    │   ├── raw
*    │   └── processed
     ├── docs/
*    ├── figs/
     ├── Python/
*    ├── outputs/
*    └── README.md
```

## Instructions

In order to reproduce the results presented in the project report, located in the docs folder, first you can clone the repository or download it, then run the file "01_inference_unvaccined_model.py", it takes less than an hour and crate the results for the unvaccinated scenario in the outputs folder, then the file "02_inference_vaccined_model.py" generate the results for the vaccinated scenario in the same folder and finally the file "03_plot_samples.py" creates the figures in the figs folder.

## Requirements
### Libraries
- tqdm
- Numpy
- Scipy
- Pandas
- Matplotlib

### Storage
You need more than 2Gb to store the results.
