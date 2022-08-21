# Final Project Computational Methods - ICTP-SAIFR/Serrapilheira

# Analysis of the impact of vaccination on the dynamics of COVID-19 in Colombia.

In this project you can find an analysis of the dynamics of the disease from mathematical modeling in Python.

## Project structure

```
project/
*    ├── data/
*    │   ├── raw
*    │   |    ├── Data.csv
*    |   |    └── MUestimates_all_locations_1.xlsx
*    │   └── processed
*    ├── docs/
*    |    ├── Covid_col.bib
*    |    ├── report.html
*    |    └── report.Rmd
*    ├── figs/
*    |    ├── fitting_model.png
*    |    ├── inference_parameters.png
*    |    ├── inference_vaccine.png
*    |    ├── state_variables_unvaccines.png
*    |    ├── state_variables_vaccines.png
*    |    └── model.png
*    ├── functions/
*    |    ├── adjust_cases.py
*    |    ├── model_agg.py
*    |    ├── utils_inference.py
*    |    └── utils_plotting.py
*    ├── outputs/
*    ├── Python/
*    |    ├── 01_inference_unvaccine_model.py
*    |    ├── 02_inference_vaccine_model.py
*    |    └── 03_plot_samples.py
*    └── README.md
```

## Instructions

In order to reproduce the results presented in the project report, located in the docs folder, first you can clone the repository or download it, then run the file "01_inference_unvaccined_model.py", it takes less than an hour and create the results for the unvaccinated scenario in the outputs folder, then the file "02_inference_vaccined_model.py" generate the results for the vaccinated scenario in the same folder and finally the file "03_plot_samples.py" creates the figures in the figs folder.

## Requirements
### Libraries
- tqdm
- Numpy
- Scipy
- Pandas
- Matplotlib

### Storage
You need more than 2Gb to store the results.
