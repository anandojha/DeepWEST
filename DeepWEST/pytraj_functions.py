import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import pytraj as pt
import mdtraj as md
import numpy as np
import scipy
import os

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


"""
# Download the reference PDB to be used when the .nc file is without solvent. Otherwise use the prmtop file
command = "curl -O http://ftp.imp.fu-berlin.de/pub/cmb-data/alanine-dipeptide-nowater.pdb"
os.system(command)
command = "mv alanine-dipeptide-nowater.pdb ref_system.pdb"
os.system(command)
get_heavy_atoms_without_solvent(traj="system_final.nc", top="ref_system.pdb", 
                                      traj_array="heavy_atoms_pt.txt", stop = 100, 
                                      stride = 10)
get_psi_phi_rad_without_solvent(index_phi=[5, 7, 9, 15], index_psi=[7, 9, 15, 17], 
                                 traj="system_final.nc", top="ref_system.pdb", 
                                 phi_psi_array="phi_psi_pt.txt",start = 0, stop = 100,
                                 stride = 10)
traj_whole_pt = np.loadtxt("heavy_atoms_pt.txt")
print(traj_whole_pt.shape)
dihedral_pt = np.loadtxt("phi_psi_pt.txt")
print(dihedral_pt.shape)
"""
