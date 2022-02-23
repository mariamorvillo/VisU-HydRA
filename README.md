# VisUQ-HydRA
Compute the estimation of the concentration of a pollutant at an environmentally sensitive target and its uncertainty. Use the output data for your risk analysis in heterogeneus hydrogeological systems.  

This computational tool box links various components relevant for the estimation of the concentration of a pollutant at an environmentally
sensitive target and its uncertainty. The computational framework builds upon existing computational tools such as HYDRO_GEN [[1]](#1),FloPy  [[2]](#2), and a 
GPU-based random walk particle tracking code [[3]](#3). The goal is to provide a ``user friendly'' platform that generate outputs that are relevant to risk analysis in groundwater systems.

# What you need
 `Tutorial_MC_F&T.ipynb` is the Jupyter Notebook including all the main scripts necessary for running the enrire workflow made available by the here proposed toolbox. Each Notebook code cell execute a different section of the whole modeling framework. Additional files, as executables and python files including function script, are needed to run each of those code cells. Those files are collected in different folders:
 
- KFields_Generator: folder containing the files related to the hydraulic conductivity fields generation.
- FlowSimulation: folder containing the files related to the flow simulations.
- TransportSimulation: folder containing the files related to the transport simulations.
- UQRA: folder containing the files related to the risk analysis and uncertainty quantification.

# How to run VisUQ-HydRA
As explained by the Markdown cells in the Jupyter notebook and in the **What you need** section, to run each code cell you need certain files. Create a folder on your computer in which you need to include the Jupyter Notebook, all the files included in the folders described above and a the Image folder, to visualize the graphycal eplanations included in the Noteboook. 

## References
<a id="1">[1]</a> 
Bellin, A. and Rubin, Y. (1996). 
Hydro_gen: A spatially distributed random field generator for correlated properties. 
Stochastic Hydrology and Hydraulics 10, 253–278

<a id="2">[2]</a> 
Bakker, M., Post, V., Langevin, C. D., Hughes, J. D., White, J., Starn, J., et al. (2016). 
Scripting modflow model development using python and flopy. 
Groundwater 54, 733–739

<a id="3">[3]</a> 
Rizzo, C. B., Nakano, A., and de Barros, F. P. J. (2019).
Par2: Parallel random walk particle tracking method for solute transport in porous media.
Computer Physics Communications 239, 265–271

