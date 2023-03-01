### Motivation

The fabrication of these devices resulted with tens of data files (`.txt`) by the end of every round. 
The extraction of all the electrical parameters was traditionally performed by manually analyzing every file: the mobility and threshold voltage require selecting the right range of data for a linear regression, and the parameters are calculated from the slope and the intercept.
This job was quite repetitive and time-consuming, so I developed this code to automate the extraction of the data. Using it I was able to focus my time on the experiments without having to worry about the data analysis part.

### Summary
In this repository you can find different Python projects I coded while completing my Bachelor's theses at [e-MolMat](https://molecularelectronics.icmab.es/) fabricating thin organic field-effect transistors (OFETs) using an innovative thin film printing technique.
For the Physics thesis, I studied the following electrical properties of the fabricated transistors:
- Electron mobility ($\mu$)
- Threshold voltage ($V_{th}$)
- On/off ratio
- Subthreshold swing (SS)
- Number of traps ($N$)

The file responsible for the extraction of the mobility, is [AutoPlot_mob_and_Vth_inTextFile.py](AutoPlot_mob_and_Vth_inTextFile.py) and it studies linear regressions of the data in every file optimizing the range of data selected and initial point in order to maximize the parameter $R^2$. 
This task is the least efficient and was used in the different folder with the data to analyze. It creates a `.txt` file called `analysis` with the mobility and thresvold voltage, including the propagated uncertainty. 
The file also offers the possibility to plot the data and regression, which allows the fast visualization of the data in the text file. This job can also be performed by the file [mob_vgs_plot.py](mob_vgs_plot.py), if only the visualization of data is desired, this code is much faster to execute, bacause it can read all folders and subfolder and plot data in all the files in them, without having to optimize a linear regression in each of the files.

The file [electrical_characterization_pristine.py](electrical_characterization_pristine.py) is very fast and powerful but requires the data to be organized in folders named after the different parameters studied.
In the case of my project, I was interested in studying the different values of the aforementioned parameters as a function of the molecule used as a semiconductor, the deposition speed (fabrication parameter), and the transistor channel length and direction with respect to the thin film printing in the fabrication.
This is why after running the file with such folders (and data files in them) a `.txt` file is created with a table format including all the electrical parameters for the different parameters regarding the fabrication and device architecture.
