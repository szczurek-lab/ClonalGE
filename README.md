# ClonalGE
This in an extention of Tumoroscope model [1]. It adds one more latent variable (B<sub>kg</sub> - average expression of gene g per one cell of clone k), one observed variable (Y<sub>sg</sub> - gene expression of gene g in spot s) and several new hyperparameters.

There are 3 main classes:
* `clonalGE.py`
* `simulation.py`
* `visualization.py`

and additional scripts, including:
* `main_diff_config.py` and `run_main_diff_config.sh` - for running ClonalGE on simulated data
* `generate_simulations.py` and `generate_simulations.sh` - for generating the simulated data
* `main_real_1000.py` and `run_main_real_1000.sh` - for running ClonalGE on the real data

The directory `configs` contains json files describing 3 different setups used to generate the simulations.  
The directory `prostate_data_configs` contains files necessary for running models on the real data (prostate cancer dataset [2])

[1] Shafighi, Shadi, et al. "Integrative spatial and genomic analysis of tumor heterogeneity with Tumoroscope." Nature Communications 15.1 (2024): 9343.  
[2] Berglund, Emelie, et al. "Spatial maps of prostate cancer transcriptomes reveal an unexplored landscape of heterogeneity." *Nature communications* 9.1 (2018): 1-13.
