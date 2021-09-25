from sklearn.mixture import GaussianMixture
import matplotlib.gridspec as gridspec
from biopandas.pdb import PandasPdb
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import stats
import tensorflow as tf
import seaborn as sns
import pandas as pd
import pytraj as pt
import mdtraj as md
import numpy as np
import itertools
import fnmatch
import shutil
import scipy
import os
import re

################ Arbitrary Functions ################

def gen_minimized_pdbs_1igd(
    dirname, 
    pad = 10, 
    Na = 2, 
    sim_steps = 10000
):

    current_pwd = os.getcwd()
    target_pwd = current_pwd + "/" + dirname
    os.chdir(target_pwd)
    files = os.listdir(".")
    file_to_find = "*.pdb"
    pdb_list = []
    for x in files:
        if fnmatch.fnmatch(x, file_to_find):
            pdb_list.append(x)
    for i in pdb_list :   
        ref_pdb = i
        leap_file = "1igd.leap"
        line_1 = "source leaprc.protein.ff14SB"
        line_2 = "source leaprc.water.tip3p"
        line_3 = "set default FlexibleWater on"
        line_4 = "set default PBRadii mbondi2"
        line_5 = "pdb = loadpdb " + ref_pdb
        line_6 = "charge pdb"
        line_7 = "solvateBox pdb TIP3PBOX " + str(pad)
        line_8 = "addions2 pdb Na+ " + str(Na)
        line_9 = "charge pdb"
        line_10 = "saveamberparm pdb " + ref_pdb[:-4] + ".prmtop " + ref_pdb[:-4] + ".inpcrd"
        line_11 = "savepdb pdb " + ref_pdb[:-4] + "_solvated.pdb"
        line_12 = "quit"
        with open(leap_file, "w") as f:
            f.write("    " + "\n")
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
        command = "tleap -f " + leap_file
        os.system(command)
        command = "rm -rf leap.log 1igd.leap"
        os.system(command)
        command = "mv " + ref_pdb + " " + " " + ref_pdb[:-4] + "_unsolvated.pdb"
        os.system(command)
        command = "mv " + ref_pdb[:-4] + "_solvated.pdb" + " " + ref_pdb
        os.system(command)
        prmtopfile = ref_pdb[:-4] + ".prmtop"
        inpcrdfile = ref_pdb[:-4] + ".inpcrd"
        sim_output = ref_pdb[:-4] + ".out"
        save_pdb = ref_pdb[:-4] + "_minimized.pdb"
        prmtop = simtk.openmm.app.AmberPrmtopFile(prmtopfile)
        inpcrd = simtk.openmm.app.AmberInpcrdFile(inpcrdfile)
        system = prmtop.createSystem()
        integrator = simtk.openmm.LangevinIntegrator(300 * simtk.unit.kelvin, 1 / simtk.unit.picosecond, 0.002 * simtk.unit.picoseconds)
        simulation = simtk.openmm.app.Simulation(prmtop.topology, system, integrator)
        simulation.context.setPositions(inpcrd.positions)
        simulation.reporters.append(simtk.openmm.app.PDBReporter(sim_output, sim_steps / 10))
        simulation.reporters.append(simtk.openmm.app.PDBReporter(save_pdb, sim_steps))
        simulation.reporters.append(simtk.openmm.app.StateDataReporter(stdout, reportInterval=int(sim_steps / 10), step=True, potentialEnergy=True, temperature=True))
        print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
        simulation.minimizeEnergy()
        print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
        simulation.step(sim_steps)
    files = os.listdir(".")
    file_to_find = "*_minimized.pdb"
    minimized_pdbs = []
    for x in files:
        if fnmatch.fnmatch(x, file_to_find):
            minimized_pdbs.append(x)
    command = "rm -rf minimized_pdbs"
    os.system(command)
    command = "mkdir minimized_pdbs"
    os.system(command)
    for i in minimized_pdbs:
        command = "mv " + i + " " + "minimized_pdbs"
        os.system(command)
    minimized_pwd = target_pwd + "/" + "minimized_pdbs"
    os.chdir(minimized_pwd)
    for i in minimized_pdbs:
        command = "mv " + i + " " + i[:-14] + ".pdb"
        os.system(command)
    os.chdir(current_pwd)

def create_1igd_for_md(
    url = "https://files.rcsb.org/download/1IGD.pdb", 
    ref_pdb = "1igd.pdb", 
    pad = 10, 
    Na = 2
):

    command = "curl -O " + url
    os.system(command)
    command = "mv 1IGD.pdb " + ref_pdb
    os.system(command)
    command = "pdb4amber -i " + ref_pdb + " -o " + "intermediate.pdb" + " --dry"
    os.system(command)
    command ="rm -rf *nonprot* *renum* *sslink* *_water* "
    os.system(command)
    file1 = open("intermediate.pdb", "r")
    file2 = open(ref_pdb, "w")
    for line in file1.readlines():
        if "REMARK" not in line:
            file2.write(line) 
    file1.close()
    file2.close()
    file1 = open(ref_pdb, "r")
    file2 = open("intermediate.pdb", "w")
    for line in file1.readlines():
        if "CRYST" not in line:
            file2.write(line) 
    file1.close()
    file2.close()
    file1 = open("intermediate.pdb", "r")
    file2 = open(ref_pdb, "w")
    for line in file1.readlines():
        if "TER" not in line:
            file2.write(line) 
    file1.close()
    file2.close()
    command = "rm -rf intermediate.pdb"
    os.system(command)
    leap_file = "1igd.leap"
    line_1 = "source leaprc.protein.ff14SB"
    line_2 = "source leaprc.water.tip3p"
    line_3 = "set default FlexibleWater on"
    line_4 = "set default PBRadii mbondi2"
    line_5 = "pdb = loadpdb " + ref_pdb
    line_6 = "charge pdb"
    line_7 = "solvateBox pdb TIP3PBOX " + str(pad)
    line_8 = "addions2 pdb Na+ " + str(Na)
    line_9 = "charge pdb"
    line_10 = "saveamberparm pdb " + ref_pdb[:-4] + ".prmtop " + ref_pdb[:-4] + ".inpcrd"
    line_11 = "savepdb pdb " + ref_pdb[:-4] + "_solvated.pdb"
    line_12 = "quit"
    with open(leap_file, "w") as f:
        f.write("    " + "\n")
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
    command = "tleap -f " + leap_file
    os.system(command)
    command = "rm -rf leap.log 1igd.leap"
    os.system(command)
    command = "mv 1igd_solvated.pdb system_final.pdb"
    os.system(command)
    command = "mv 1igd.prmtop system_final.prmtop"
    os.system(command)
    command = "mv 1igd.inpcrd system_final.inpcrd"
    os.system(command)

def get_non_H_unique_atoms(traj):
    ppdb = PandasPdb()
    ppdb.read_pdb(traj)
    no_host_atoms = ppdb.df["ATOM"].shape[0]
    df = ppdb.df["ATOM"]
    unique_list = list(df.atom_name.unique())
    for word in unique_list[:]:
        if word.startswith("H"):
            unique_list.remove(word)
    return(unique_list)

def get_traj_pdb_from_nc_solvent(
    traj,
    top,
    traj_pdb,
    start=0,
    stop=100000,
    stride=1,
):

    """
    Converts NETCDF trajectory data with solvent to a multiPDB trajectory data
    """

    trajec = md.load(traj, top=top)
    trajec = trajec.remove_solvent()
    trajec = trajec[start:stop:stride]
    print(trajec)
    trajec.save_pdb(traj_pdb, force_overwrite=True)
    fin = open(traj_pdb, "rt")
    fout = open("intermediate_I.txt", "wt")
    for line in fin:
        fout.write(line.replace("ENDMDL", "END"))
    fin.close()
    fout.close()
    file1 = open("intermediate_I.txt", "r")
    file2 = open("intermediate_II.txt", "w")
    for line in file1.readlines():
        if not (line.startswith("TER")):
            file2.write(line)
    file2.close()
    file1.close()
    file1 = open("intermediate_II.txt", "r")
    file2 = open("intermediate_III.txt", "w")
    for line in file1.readlines():
        if not (line.startswith("REMARK")):
            file2.write(line)
    file2.close()
    file1.close()
    file1 = open("intermediate_III.txt", "r")
    file2 = open("intermediate_IV.txt", "w")
    for line in file1.readlines():
        if not (line.startswith("MODEL")):
            file2.write(line)
    file2.close()
    file1.close()
    file1 = open("intermediate_IV.txt", "r")
    file2 = open("intermediate_V.txt", "w")
    for line in file1.readlines():
        if not (line.startswith("CRYST")):
            file2.write(line)
    file2.close()
    file1.close()
    file1 = open("intermediate_V.txt", "r")
    file2 = open("intermediate_VI.txt", "w")
    for line in file1.readlines():
        if not (line.startswith("CONECT")):
            file2.write(line)
    file2.close()
    file1.close()
    with open("intermediate_VI.txt") as f1:
        lines = f1.readlines()
    with open(traj_pdb, "w") as f2:
        f2.writelines(lines[:-1])
    command = "rm -rf intermediate_I.txt intermediate_II.txt intermediate_III.txt intermediate_IV.txt intermediate_V.txt intermediate_VI.txt"
    os.system(command)

def get_alad_traj_pdb_from_nc(
    traj,
    ref_pdb="reference.pdb",
    start=0,
    stop=100000,
    stride=1,
    traj_pdb="alad_multi.pdb",
):

    """
    Converts NETCDF trajectory data without solvent to a multiPDB trajectory data
    """
    command = "curl -O http://ftp.imp.fu-berlin.de/pub/cmb-data/alanine-dipeptide-nowater.pdb"
    os.system(command)
    command = "mv alanine-dipeptide-nowater.pdb " + ref_pdb
    os.system(command)
    topology = md.load(ref_pdb).topology
    print(topology)
    trajec = md.load(traj, top=ref_pdb)
    trajec = trajec[start:stop:stride]
    print(trajec)
    trajec.save_pdb(traj_pdb, force_overwrite=True)
    command = "rm -rf " + ref_pdb
    os.system(command)
    fin = open(traj_pdb, "rt")
    fout = open("intermediate_I.txt", "wt")
    for line in fin:
        fout.write(line.replace("ENDMDL", "END"))
    fin.close()
    fout.close()
    file1 = open("intermediate_I.txt", "r")
    file2 = open("intermediate_II.txt", "w")
    for line in file1.readlines():
        if not (line.startswith("TER")):
            file2.write(line)
    file2.close()
    file1.close()
    file1 = open("intermediate_II.txt", "r")
    file2 = open("intermediate_III.txt", "w")
    for line in file1.readlines():
        if not (line.startswith("REMARK")):
            file2.write(line)
    file2.close()
    file1.close()
    file1 = open("intermediate_III.txt", "r")
    file2 = open("intermediate_IV.txt", "w")
    for line in file1.readlines():
        if not (line.startswith("MODEL")):
            file2.write(line)
    file2.close()
    file1.close()
    file1 = open("intermediate_IV.txt", "r")
    file2 = open("intermediate_V.txt", "w")
    for line in file1.readlines():
        if not (line.startswith("CRYST")):
            file2.write(line)
    file2.close()
    file1.close()
    file1 = open("intermediate_V.txt", "r")
    file2 = open("intermediate_VI.txt", "w")
    for line in file1.readlines():
        if not (line.startswith("CONECT")):
            file2.write(line)
    file2.close()
    file1.close()
    with open("intermediate_VI.txt") as f1:
        lines = f1.readlines()
    with open(traj_pdb, "w") as f2:
        f2.writelines(lines[:-1])
    command = "rm -rf intermediate_I.txt intermediate_II.txt intermediate_III.txt intermediate_IV.txt intermediate_V.txt intermediate_VI.txt"
    os.system(command)

def get_chignolin_traj_pdb_from_nc(
    traj,
    ref_pdb="reference.pdb",
    start=0,
    stop=100000,
    stride=1,
    traj_pdb="chig_multi.pdb",
):

    """
    Converts NETCDF trajectory data without solvent to a multiPDB trajectory data
    """
    command = "curl -O https://files.rcsb.org/download/1UAO.pdb1.gz"
    os.system(command)
    command = "gunzip 1UAO.pdb1.gz"
    os.system(command)
    command = "mv 1UAO.pdb1 " + ref_pdb
    os.system(command)
    topology = md.load(ref_pdb).topology
    print(topology)
    trajec = md.load(traj, top=ref_pdb)
    trajec = trajec[start:stop:stride]
    print(trajec)
    trajec.save_pdb(traj_pdb, force_overwrite=True)
    command = "rm -rf " + ref_pdb
    os.system(command)
    fin = open(traj_pdb, "rt")
    fout = open("intermediate_I.txt", "wt")
    for line in fin:
        fout.write(line.replace("ENDMDL", "END"))
    fin.close()
    fout.close()
    file1 = open("intermediate_I.txt", "r")
    file2 = open("intermediate_II.txt", "w")
    for line in file1.readlines():
        if not (line.startswith("TER")):
            file2.write(line)
    file2.close()
    file1.close()
    file1 = open("intermediate_II.txt", "r")
    file2 = open("intermediate_III.txt", "w")
    for line in file1.readlines():
        if not (line.startswith("REMARK")):
            file2.write(line)
    file2.close()
    file1.close()
    file1 = open("intermediate_III.txt", "r")
    file2 = open("intermediate_IV.txt", "w")
    for line in file1.readlines():
        if not (line.startswith("MODEL")):
            file2.write(line)
    file2.close()
    file1.close()
    file1 = open("intermediate_IV.txt", "r")
    file2 = open("intermediate_V.txt", "w")
    for line in file1.readlines():
        if not (line.startswith("CRYST")):
            file2.write(line)
    file2.close()
    file1.close()
    file1 = open("intermediate_V.txt", "r")
    file2 = open("intermediate_VI.txt", "w")
    for line in file1.readlines():
        if not (line.startswith("CONECT")):
            file2.write(line)
    file2.close()
    file1.close()
    with open("intermediate_VI.txt") as f1:
        lines = f1.readlines()
    with open(traj_pdb, "w") as f2:
        f2.writelines(lines[:-1])
    command = "rm -rf intermediate_I.txt intermediate_II.txt intermediate_III.txt intermediate_IV.txt intermediate_V.txt intermediate_VI.txt"
    os.system(command)


def create_heavy_atom_xyz(
    traj, ref_pdb, heavy_atoms_array, start=0, stop=500000, stride=1
):
    # Download the reference PDB to be used when the .nc file is without solvent. Otherwise, use
    # the prmtop file
    command = "curl -O http://ftp.imp.fu-berlin.de/pub/cmb-data/alanine-dipeptide-nowater.pdb"
    os.system(command)
    command = "mv alanine-dipeptide-nowater.pdb " + ref_pdb
    os.system(command)
    topology = md.load(ref_pdb).topology
    print(topology)
    df, bonds = topology.to_dataframe()
    heavy_indices = list(df[df["element"] != "H"].index)
    print(heavy_indices)
    trajec = md.load(traj, top=ref_pdb)
    trajec = trajec[start:stop:stride]
    print(trajec)
    trajec = trajec.atom_slice(atom_indices=heavy_indices)
    print(trajec)
    trajec_xyz = trajec.xyz * 10
    print(trajec_xyz.shape)
    trajec_xyz = trajec_xyz.reshape(
        (trajec.xyz.shape[0], trajec.xyz.shape[1] * trajec.xyz.shape[2])
    )
    print(trajec_xyz.shape)
    np.savetxt(heavy_atoms_array, trajec_xyz)
    

def create_phi_psi(traj, ref_pdb, phi_psi_txt, start=0, stop=500000, stride=1):
    trajec = md.load(traj, top=ref_pdb)
    trajec = trajec[start:stop:stride]
    phi = md.compute_phi(trajec)
    phi = phi[1]  # 0:indices, 1:phi angles
    print(phi.shape)
    psi = md.compute_psi(trajec)
    psi = psi[1]  # 0:indices, 1:phi angles
    print(psi.shape)
    phi_psi = np.array([list(x) for x in zip(phi, psi)])
    print(phi_psi.shape)
    phi_psi = phi_psi.reshape(
        (phi_psi.shape[0], phi_psi.shape[1] * phi_psi.shape[2])
    )
    print(phi_psi.shape)
    np.savetxt(phi_psi_txt, phi_psi)


def create_phi_psi_solvent(
    traj, top, phi_psi_txt, start=0, stop=500000, stride=1
):
    trajec = md.load(traj, top=top)
    trajec = trajec.remove_solvent()
    trajec = trajec[start:stop:stride]
    phi = md.compute_phi(trajec)
    phi = phi[1]  # 0:indices, 1:phi angles
    print(phi.shape)
    psi = md.compute_psi(trajec)
    psi = psi[1]  # 0:indices, 1:phi angles
    print(psi.shape)
    phi_psi = np.array([list(x) for x in zip(phi, psi)])
    print(phi_psi.shape)
    phi_psi = phi_psi.reshape(
        (phi_psi.shape[0], phi_psi.shape[1] * phi_psi.shape[2])
    )
    print(phi_psi.shape)
    np.savetxt(phi_psi_txt, phi_psi)


def fix_cap_remove_ace(pdb_file):

    """
    Removes the H atoms of the capped ACE residue.

    """

    remove_words = [
        "H1  ACE",
        "H2  ACE",
        "H3  ACE",
        "H31 ACE",
        "H32 ACE",
        "H33 ACE",
    ]
    with open(pdb_file) as oldfile, open("intermediate.pdb", "w") as newfile:
        for line in oldfile:
            if not any(word in line for word in remove_words):
                newfile.write(line)
    command = "rm -rf " + pdb_file
    os.system(command)
    command = "mv intermediate.pdb " + pdb_file
    os.system(command)


def fix_cap_replace_ace(pdb_file):

    """
    Replaces the alpha carbon atom of the
    capped ACE residue with a standard name.

    """

    fin = open(pdb_file, "rt")
    data = fin.read()
    data = data.replace("CA  ACE", "CH3 ACE")
    data = data.replace("C   ACE", "CH3 ACE")
    fin.close()
    fin = open(pdb_file, "wt")
    fin.write(data)
    fin.close()


def add_dihedral_input(traj_whole, dihedral):
    return np.hstack((traj_whole, dihedral))

################ Arbitrary Functions ################

################ Pytraj Functions ################

def get_heavy_atoms_without_solvent(traj, top, traj_array, start = 0, stop = 10000, stride = 1):
    trajectory = pt.iterload(traj, top, frame_slice = (start, stop, stride))
    print(trajectory)
    uniq_resids = list(set(residue.name for residue in trajectory.top.residues))
    print(uniq_resids)
    uniq_resids_no_solvent = []
    for i in uniq_resids:
        if i != "WAT":
            uniq_resids_no_solvent.append(i)
    print(uniq_resids_no_solvent)
    str_input = ",".join(uniq_resids_no_solvent)
    str_input = ":" + str_input
    print(str_input)
    traj_no_solvent = trajectory[str_input]
    print(traj_no_solvent)
    resid_list = [atom.name for atom in traj_no_solvent.top.atoms][:]
    print(resid_list)
    h_resid_list = []
    for i in resid_list:  # assuming residues does not include Hafnium,
        # Hassium, Helium & Holmium
        if i.startswith("H"):
            h_resid_list.append(i)
    print(h_resid_list)
    non_h_resid_list = [x for x in resid_list if x not in h_resid_list]
    print(non_h_resid_list)
    str_input_ = ",".join(non_h_resid_list)
    str_input_ = "@" + str_input_
    print(str_input_)
    traj_no_solvent_no_h = traj_no_solvent[str_input_]
    print(traj_no_solvent_no_h)
    xyzfile = "system_heavy.xyz"
    traj_no_solvent_no_h.save(xyzfile, overwrite=True)
    data = pd.read_csv(
        xyzfile, header=None, delim_whitespace=True, skiprows=[0]
    )
    data = data[[1, 2, 3]]
    data.columns = ["x_coor", "y_coor", "z_coor"]
    print(data.head())
    array_data = data.to_numpy()
    print(array_data.shape)
    print(array_data)
    dim1 = int(array_data.shape[0] / len(non_h_resid_list))
    dim2 = int(array_data.shape[1] * len(non_h_resid_list))
    array_data_heavy = array_data.reshape((dim1, dim2))
    print(array_data_heavy.shape)
    print(array_data_heavy)
    np.savetxt(traj_array, array_data_heavy)
    command = "rm -rf " + xyzfile
    os.system(command)
    
def get_psi_phi_rad_without_solvent(index_phi, index_psi, traj, top, 
                                    phi_psi_array, start = 0, stop = 10000, stride = 1):
    traj = pt.iterload(traj, top, frame_slice = (start, stop, stride))
    index_phi_add = (
        "@"
        + str(index_phi[0])
        + " @"
        + str(index_phi[1])
        + " @"
        + str(index_phi[2])
        + " @"
        + str(index_phi[3])
    )
    print(index_phi_add)
    index_psi_add = (
        "@"
        + str(index_psi[0])
        + " @"
        + str(index_psi[1])
        + " @"
        + str(index_psi[2])
        + " @"
        + str(index_psi[3])
    )
    print(index_psi_add)
    phi = pt.dihedral(traj, index_phi_add)
    phi_rad = np.array([np.deg2rad(i) for i in phi])
    psi = pt.dihedral(traj, index_psi_add)
    psi_rad = np.array([np.deg2rad(i) for i in psi])
    phi_psi = np.array([list(x) for x in zip(phi_rad, psi_rad)])
    np.savetxt(phi_psi_array, phi_psi)
    
    
def get_psi_phi_degrees_without_solvent(index_phi, index_psi, traj, top, 
                                    phi_psi_array, start = 0, stop = 10000, stride = 1):
    traj = pt.iterload(traj, top, frame_slice = (start, stop, stride))
    index_phi_add = (
        "@"
        + str(index_phi[0])
        + " @"
        + str(index_phi[1])
        + " @"
        + str(index_phi[2])
        + " @"
        + str(index_phi[3])
    )
    print(index_phi_add)
    index_psi_add = (
        "@"
        + str(index_psi[0])
        + " @"
        + str(index_psi[1])
        + " @"
        + str(index_psi[2])
        + " @"
        + str(index_psi[3])
    )
    print(index_psi_add)
    phi = pt.dihedral(traj, index_phi_add)
    psi = pt.dihedral(traj, index_psi_add)
    phi_psi = np.array([list(x) for x in zip(phi, psi)])
    np.savetxt(phi_psi_array, phi_psi)

def get_heavy_atoms_with_solvent(traj, top, traj_array):
    traj = pt.iterload(traj, top)
    print(traj)
    uniq_resids = list(set(residue.name for residue in traj.top.residues))
    print(uniq_resids)
    uniq_resids_no_solvent = []
    for i in uniq_resids:
        if i != "WAT":
            uniq_resids_no_solvent.append(i)
    print(uniq_resids_no_solvent)
    str_input = ",".join(uniq_resids_no_solvent)
    str_input = ":" + str_input
    print(str_input)
    traj_no_solvent = traj[str_input]
    print(traj_no_solvent)
    resid_list = [atom.name for atom in traj_no_solvent.top.atoms][:]
    print(resid_list)
    h_resid_list = []
    for i in resid_list:  # assuming residues does not include Hafnium,
        # Hassium, Helium & Holmium
        if i.startswith("H"):
            h_resid_list.append(i)
    print(h_resid_list)
    non_h_resid_list = [x for x in resid_list if x not in h_resid_list]
    print(non_h_resid_list)
    str_input_ = ",".join(non_h_resid_list)
    str_input_ = "@" + str_input_
    print(str_input_)
    traj_no_solvent_no_h = traj_no_solvent[str_input_]
    print(traj_no_solvent_no_h)
    xyzfile = "system_heavy.xyz"
    traj_no_solvent_no_h.save(xyzfile, overwrite=True)
    data = pd.read_csv(
        xyzfile, header=None, delim_whitespace=True, skiprows=[0]
    )
    data = data[[1, 2, 3]]
    data.columns = ["x_coor", "y_coor", "z_coor"]
    print(data.head())
    array_data = data.to_numpy()
    print(array_data.shape)
    print(array_data)
    dim1 = int(array_data.shape[0] / len(non_h_resid_list))
    dim2 = int(array_data.shape[1] * len(non_h_resid_list))
    array_data_heavy = array_data.reshape((dim1, dim2))
    print(array_data_heavy.shape)
    print(array_data_heavy)
    np.savetxt(traj_array, array_data_heavy)
    command = "rm -rf " + xyzfile
    os.system(command)


def get_psi_phi_degrees_with_solvent(index_phi, index_psi, traj, top, phi_psi_array):
    traj = pt.load(traj, top)
    index_phi_add = (
        "@"
        + str(index_phi[0])
        + " @"
        + str(index_phi[1])
        + " @"
        + str(index_phi[2])
        + " @"
        + str(index_phi[3])
    )
    print(index_phi_add)
    index_psi_add = (
        "@"
        + str(index_psi[0])
        + " @"
        + str(index_psi[1])
        + " @"
        + str(index_psi[2])
        + " @"
        + str(index_psi[3])
    )
    print(index_psi_add)
    phi = pt.dihedral(traj, index_phi_add)
    psi = pt.dihedral(traj, index_psi_add)
    phi_psi = np.array([list(x) for x in zip(phi, psi)])
    np.savetxt(phi_psi_array, phi_psi)


def get_psi_phi_rad_with_solvent(index_phi, index_psi, traj, top, phi_psi_array):
    traj = pt.load(traj, top)
    index_phi_add = (
        "@"
        + str(index_phi[0])
        + " @"
        + str(index_phi[1])
        + " @"
        + str(index_phi[2])
        + " @"
        + str(index_phi[3])
    )
    print(index_phi_add)
    index_psi_add = (
        "@"
        + str(index_psi[0])
        + " @"
        + str(index_psi[1])
        + " @"
        + str(index_psi[2])
        + " @"
        + str(index_psi[3])
    )
    print(index_psi_add)
    phi = pt.dihedral(traj, index_phi_add)
    phi_rad = np.array([np.deg2rad(i) for i in phi])
    psi = pt.dihedral(traj, index_psi_add)
    psi_rad = np.array([np.deg2rad(i) for i in psi])
    phi_psi = np.array([list(x) for x in zip(phi_rad, psi_rad)])
    np.savetxt(phi_psi_array, phi_psi)

################ Pytraj Functions ################

