DeepWEST
==============================

Deep Learning Analysis of Molecular Dynamics Trajectories for Weighted Ensemble simulations


## Installation and Setup Instructions :
* Make sure [anaconda3](https://www.anaconda.com/) is installed on the local machine. Go to the  [download](https://www.anaconda.com/products/individual) page of anaconda3 and install the latest version of anaconda3.
* Create a new conda environment with python = 3.7 and install the package with the following commands in the terminal: 
```bash
conda create -n deepwest python=3.7

conda activate deepwest

conda install -c conda-forge matplotlib scipy jupyterlab pandas tensorflow=2.2.0 mdtraj scikit-learn seaborn pytorch openmm=7.5.0 biopandas ambertools

from biopandas.pdb import PandasPdb
import pandas as pd
import argparse
import os
import re
parser = argparse.ArgumentParser(description='PDB file name')
parser.add_argument('-p','--pdb', help='Input PDB File', required=True)
args = parser.parse_args()
pdb_file = args.pdb
################################################################################################################################################
salt_conc = 150
resids = ["ALA", "ARG", "ASN", "ASP", "ASX", "CYS", "GLN", "GLU", "GLX", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "ACE", "Y2P", "HIE", "NME", "HID", "WAT", "ATP", "GDP"]
weights_Da = [89, 174, 132, 133, 133, 121, 146, 147, 147, 75, 155, 131, 131, 146, 149, 165, 115, 105, 119, 204, 181, 117, 43, 261, 155, 31, 155, 0, 0, 0]
resids_weights = pd.DataFrame({'resids': resids, 'weights_Da': weights_Da})
################################################################################################################################################
line_1 = "loadamberparams atp.frcmod"
line_2 = "loadamberparams gdp.frcmod"
line_3 = "loadamberprep atp.in"
line_4 = "loadamberprep gdp.in"
line_5 = "source leaprc.protein.ff19SB"
line_6 = "source leaprc.gaff"
line_7 = "source leaprc.water.tip4pew"
line_8 = "set default FlexibleWater on"
line_9 = "set default PBRadii mbondi2"
line_10 = "WAT = T4E"
line_11 = "HOH = T4E"
line_12 = "loadAmberParams frcmod.ionsjc_tip4pew"
line_13 = "loadAmberParams frcmod.tip4pew"
line_14 = "pdb = loadpdb " + pdb_file
line_15 = "charge pdb"
line_16 = "solvatebox pdb TIP4PEWBOX 12"
line_17 = "savepdb pdb system_salt.pdb"
line_18 = "quit"

with open('input_salt.leap', 'w') as f:
    f.write(line_1 + "\n")
    f.write(line_2 + "\n")
    f.write(line_3 + "\n")
    f.write(line_4 + "\n")
    f.write(line_5 + "\n")
    f.write(line_6 + "\n")
    f.write(line_7 + "\n")
    f.write(line_8 + "\n")
    f.write(line_9 + "\n")
    f.write(line_10 + "\n")
    f.write(line_11 + "\n")
    f.write(line_12 + "\n")
    f.write(line_13 + "\n")
    f.write(line_14 + "\n")
    f.write(line_15 + "\n")
    f.write(line_16 + "\n")
    f.write(line_17 + "\n")
    f.write(line_18 + "\n")
os.system("tleap -f input_salt.leap > input_salt.out")
################################################################################################################################################
ppdb = PandasPdb()
ppdb.read_pdb("system_salt.pdb")
df = ppdb.df["ATOM"]
df_grouped = df.groupby('residue_name')['residue_number'].nunique()
pdb_resid_list = df_grouped.index.to_list()
#print(pdb_resid_list)
pdb_resid_frequency_list = df_grouped.values.tolist()
#print(pdb_resid_frequency_list)
weights_list = []
for i in pdb_resid_list:
    weights_list.append(resids_weights["weights_Da"][resids_weights.loc[resids_weights.isin([i]).any(axis=1)].index.tolist()[0]])
#print(weights_list)
protein_weight = sum([a*b for a,b in zip(weights_list,pdb_resid_frequency_list)])/1000
#print(protein_weight)
with open("system_salt.pdb","r") as f:
    water_lines = []   
    for line in f:
        if "EPW" in line:
            water_lines.append(line)
no_water = len(water_lines)
with open("input_salt.out","r") as file: 
    lines = file.readlines()
    for line in lines:
        if line.startswith("Total perturbed charge:"):
            charge = float(round(float(re.findall(r'[-+]?\d+[,.]?\d*',line)[0])))
################################################################################################################################################
os.system("rm -rf input_salt.leap leap.log input_salt.out")
print("Please go the website " , "https://www.phys.ksu.edu/personal/schmit/SLTCAP/SLTCAP.html" ,
      " and fill in with the following details to get the number of ions required for the system")
print("Protein mass (kDa):", protein_weight)
print("Solution salt concentration (mM/l):", salt_conc)
print("Net charge of solutes (proton charge units):", int(charge))
print("Number of water molecules:", no_water)
################################################################################################################################################
