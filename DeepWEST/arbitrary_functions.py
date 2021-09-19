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


def fix_cap_remove_nme(pdb_file):

    """
    Removes the H atoms of the capped NME residue.

    """

    remove_words = [
        "H1  NME",
        "H2  NME",
        "H3  NME",
        "H31 NME",
        "H32 NME",
        "H33 NME",
    ]
    with open(pdb_file) as oldfile, open("intermediate.pdb", "w") as newfile:
        for line in oldfile:
            if not any(word in line for word in remove_words):
                newfile.write(line)
    command = "rm -rf " + pdb_file
    os.system(command)
    command = "mv intermediate.pdb " + pdb_file
    os.system(command)




def create_westpa_dir(traj_file, top, indices):
    os.system("rm -rf westpa_dir")
    os.system("mkdir westpa_dir")
    for i in indices:
        traj_frame = md.load_frame(filename=traj_file, top=top, index=i)
        pdb_name = str(i) + ".pdb"
        pdb_path = os.path.join(os.getcwd(), "westpa_dir/" + pdb_name)
        traj_frame.save_pdb(pdb_path, force_overwrite=True)


def add_vectors(traj, top, inpcrd_file):
    trajec = md.load_frame(traj, top=top, index=0)
    x = trajec.openmm_boxes(frame=0)
    x = str(x)
    x = x.replace("Vec3", "")
    x = re.findall("\d*\.?\d+", x)
    for i in range(0, len(x)):
        x[i] = float(x[i])
    x = tuple(x)
    n = int(len(x) / 3)
    x = [x[i * n : (i + 1) * n] for i in range((len(x) + n - 1) // n)]
    vectors = ((x[0][0]) * 10, (x[1][1]) * 10, (x[2][2]) * 10)
    vectors = (
        round(vectors[0], 7),
        round(vectors[1], 7),
        round(vectors[2], 7),
    )
    new_vectors = []
    for i in vectors:
        target_len = 10
        if len(str(i)) < 10:
            i = str(i) + (10 - len(str(i))) * "0"
        else:
            i = str(i)
        new_vectors.append(i)
    new_vectors = tuple(new_vectors)
    last_line = (
        "  "
        + new_vectors[0]
        + "  "
        + new_vectors[1]
        + "  "
        + new_vectors[2]
        + "  90.0000000"
        + "  90.0000000"
        + "  90.0000000"
    )
    with open(inpcrd_file, "a+") as f:
        f.write(last_line)


def run_short_md_westpa_dir(traj, top):
    files = os.listdir(".")
    file_to_find = "*.pdb"
    pdb_list = []
    for x in files:
        if fnmatch.fnmatch(x, file_to_find):
            pdb_list.append(x)
    # Fixing capping issues in mdtraj saved pdb files
    for i in pdb_list:
        pdb_file = i
        fix_cap_remove_nme(pdb_file)
        fix_cap_replace_nme(pdb_file)
    # Saving inpcrd file from mdtraj saved pdb files
    for i in pdb_list:
        pdb_file = i
        line_1 = "source leaprc.protein.ff14SB"
        line_2 = "source leaprc.water.tip3p"
        line_3 = "set default FlexibleWater on"
        line_4 = "set default PBRadii mbondi2"
        line_5 = "pdb = loadpdb " + pdb_file
        line_6 = (
            "saveamberparm pdb "
            + pdb_file[:-4]
            + ".prmtop "
            + pdb_file[:-4]
            + ".inpcrd"
        )
        line_7 = "quit"
        with open("input.leap", "w") as f:
            f.write("    " + "\n")
            f.write(line_1 + "\n")
            f.write(line_2 + "\n")
            f.write(line_3 + "\n")
            f.write(line_4 + "\n")
            f.write(line_5 + "\n")
            f.write(line_6 + "\n")
            f.write(line_7 + "\n")
        command = "tleap -f input.leap"
        os.system(command)
        command = "rm -rf input.leap"
        os.system(command)
    files = os.listdir(".")
    file_to_find = "*.inpcrd"
    inpcrd_list = []
    for x in files:
        if fnmatch.fnmatch(x, file_to_find):
            inpcrd_list.append(x)
    for i in inpcrd_list:
        add_vectors(traj=traj, top=top, inpcrd_file=i)
    # Creating Amber MD input file
    with open("md.in", "w") as f:
        f.write("Run minimization followed by saving rst file" + "\n")
        f.write("&cntrl" + "\n")
        f.write(
            "  imin = 1, maxcyc = 10000, ntpr = 5, iwrap = 1, ntxo = 1" + "\n"
        )
        f.write("&end" + "\n")
    # Running short MD simulations to save .rst file
    for i in pdb_list:
        pdb_file = i
        command = (
            "pmemd.cuda -O -i md.in -o "
            + pdb_file[:-4]
            + ".out"
            + " -p "
            + pdb_file[:-4]
            + ".prmtop"
            + " -c "
            + pdb_file[:-4]
            + ".inpcrd"
            + " -r "
            + pdb_file[:-4]
            + ".rst"
        )
        print(command)
        os.system(command)
    # Deleting md.in file
    command = "rm -rf md.in __pycache__  leap.log mdinfo"
    os.system(command)


def create_westpa_filetree():
    files = os.listdir(".")
    file_to_find = "*.rst"
    rst_list = []
    for x in files:
        if fnmatch.fnmatch(x, file_to_find):
            rst_list.append(x)
    current_dir = os.getcwd()
    target_dir = current_dir + "/" + "bstates_rst"
    os.system("rm -rf bstates_rst")
    os.system("mkdir bstates_rst")
    for i in rst_list:
        shutil.copy(current_dir + "/" + i, target_dir + "/" + i)

    files = os.listdir(".")
    file_to_find = "*.pdb"
    pdb_list = []
    for x in files:
        if fnmatch.fnmatch(x, file_to_find):
            pdb_list.append(x)
    current_dir = os.getcwd()
    target_dir = current_dir + "/" + "bstates_pdb"
    os.system("rm -rf bstates_pdb")
    os.system("mkdir bstates_pdb")
    for i in pdb_list:
        shutil.copy(current_dir + "/" + i, target_dir + "/" + i)

    files = os.listdir(".")
    file_to_find = "*.inpcrd"
    inpcrd_list = []
    for x in files:
        if fnmatch.fnmatch(x, file_to_find):
            inpcrd_list.append(x)
    current_dir = os.getcwd()
    target_dir = current_dir + "/" + "bstates_inpcrd"
    os.system("rm -rf bstates_inpcrd")
    os.system("mkdir bstates_inpcrd")
    for i in inpcrd_list:
        shutil.copy(current_dir + "/" + i, target_dir + "/" + i)

    files = os.listdir(".")
    file_to_find = "*.prmtop"
    prmtop_list = []
    for x in files:
        if fnmatch.fnmatch(x, file_to_find):
            prmtop_list.append(x)
    current_dir = os.getcwd()
    target_dir = current_dir + "/" + "bstates_prmtop"
    os.system("rm -rf bstates_prmtop")
    os.system("mkdir bstates_prmtop")
    for i in prmtop_list:
        shutil.copy(current_dir + "/" + i, target_dir + "/" + i)

    files = os.listdir(".")
    file_to_find = "*.out"
    out_list = []
    for x in files:
        if fnmatch.fnmatch(x, file_to_find):
            out_list.append(x)
    current_dir = os.getcwd()
    target_dir = current_dir + "/" + "bstates_out"
    os.system("rm -rf bstates_out")
    os.system("mkdir bstates_out")
    for i in out_list:
        shutil.copy(current_dir + "/" + i, target_dir + "/" + i)

    files = os.listdir(".")
    file_to_find = "*.rst"
    rst_list = []
    for x in files:
        if fnmatch.fnmatch(x, file_to_find):
            rst_list.append(x)
    prob = round(1 / len(rst_list), 10)
    prob_list = [prob] * len(rst_list)
    index = []
    for i in range(len(rst_list)):
        index.append(i)
    data = [index, prob_list, rst_list]
    df = pd.DataFrame(data)
    df = df.transpose()
    df.columns = ["index", "prob", "file"]
    df.to_csv("BASIS_STATES_RST", sep="\t", index=False, header=False)

    files = os.listdir(".")
    file_to_find = "*.inpcrd"
    inpcrd_list = []
    for x in files:
        if fnmatch.fnmatch(x, file_to_find):
            inpcrd_list.append(x)
    prob = round(1 / len(inpcrd_list), 10)
    prob_list = [prob] * len(inpcrd_list)
    index = []
    for i in range(len(inpcrd_list)):
        index.append(i)
    data = [index, prob_list, inpcrd_list]
    df = pd.DataFrame(data)
    df = df.transpose()
    df.columns = ["index", "prob", "file"]
    df.to_csv("BASIS_STATES_INPCRD", sep="\t", index=False, header=False)

    files = os.listdir(".")
    file_to_find = "*.prmtop"
    prmtop_list = []
    for x in files:
        if fnmatch.fnmatch(x, file_to_find):
            prmtop_list.append(x)
    current_dir = os.getcwd()
    target_dir = current_dir + "/" + "CONFIG"
    os.system("rm -rf CONFIG")
    os.system("mkdir CONFIG")
    prmtop_file = prmtop_list[0]
    shutil.copy(
        current_dir + "/" + prmtop_file, target_dir + "/" + "system.prmtop"
    )

    command = "rm -rf *.pdb* *.inpcrd* *.prmtop* *.rst* *.out* "
    os.system(command)


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

################ Pytraj Functions ################

