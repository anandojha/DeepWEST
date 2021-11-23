from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import seaborn as sns
from math import exp
import pandas as pd
import mdtraj as md
import pickle as pk
import numpy as np
import statistics
import itertools
import fileinput
import fnmatch
import shutil
import random
import math
import os
import re


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


def fix_cap_replace_nme(pdb_file):

    """
    Replaces the alpha carbon atom of the
    capped NME residue with a standard name.

    """

    fin = open(pdb_file, "rt")
    data = fin.read()
    data = data.replace("CA  NME", "CH3 NME")
    data = data.replace("C   NME", "CH3 NME")
    fin.close()
    fin = open(pdb_file, "wt")
    fin.write(data)
    fin.close()


def prepare_alanine_dipeptide():

    """

    Prepares the alanine dipeptide system for Gaussian
    Accelerated Molecular Dynamics (GaMD) simulations.
    Downloads the pdb structure from
    https://markovmodel.github.io/mdshare/ALA2/ and
    parameterizes it using General Amber Force Field
    (GAFF).

    """

    os.system(
        "curl -O http://ftp.imp.fu-berlin.de/pub/cmb-data/alanine-dipeptide-nowater.pdb"
    )
    os.system(
        "rm -rf system_inputs"
    )  # Removes any existing directory named system_inputs
    os.system("mkdir system_inputs")  # Creates a directory named system_inputs
    cwd = os.getcwd()
    target_dir = cwd + "/" + "system_inputs"
    os.system("pdb4amber -i alanine-dipeptide-nowater.pdb -o intermediate.pdb")
    # Delete HH31, HH32 and HH33 from the ACE residue (tleap adds them later)
    remove_words = ["HH31 ACE", "HH32 ACE", "HH33 ACE"]
    with open("intermediate.pdb") as oldfile, open(
        "system.pdb", "w"
    ) as newfile:
        for line in oldfile:
            if not any(word in line for word in remove_words):
                newfile.write(line)
    os.system("rm -rf intermediate*")
    # save the tleap script to file
    with open("input_TIP3P.leap", "w") as f:
        f.write(
            """
    source leaprc.protein.ff14SB
    source leaprc.water.tip3p
    set default FlexibleWater on
    set default PBRadii mbondi2
    pdb = loadpdb system.pdb
    solvateBox pdb TIP3PBOX 15
    saveamberparm pdb system_TIP3P.prmtop system_TIP3P.inpcrd
    saveamberparm pdb system_TIP3P.parm7 system_TIP3P.rst7
    savepdb pdb system_TIP3P.pdb
    quit
    """
        )
    os.system("tleap -f input_TIP3P.leap")
    os.system("rm -rf leap.log")
    shutil.copy(
        cwd + "/" + "system_TIP3P.inpcrd",
        target_dir + "/" + "system_TIP3P.inpcrd",
    )
    shutil.copy(
        cwd + "/" + "system_TIP3P.parm7",
        target_dir + "/" + "system_TIP3P.parm7",
    )
    shutil.copy(
        cwd + "/" + "system_TIP3P.pdb", target_dir + "/" + "system_TIP3P.pdb"
    )
    shutil.copy(
        cwd + "/" + "system_TIP3P.prmtop",
        target_dir + "/" + "system_TIP3P.prmtop",
    )
    shutil.copy(
        cwd + "/" + "system_TIP3P.rst7", target_dir + "/" + "system_TIP3P.rst7"
    )
    shutil.copy(cwd + "/" + "system.pdb", target_dir + "/" + "system.pdb")
    shutil.copy(
        cwd + "/" + "alanine-dipeptide-nowater.pdb",
        target_dir + "/" + "alanine-dipeptide-nowater.pdb",
    )
    shutil.copy(
        cwd + "/" + "input_TIP3P.leap", target_dir + "/" + "input_TIP3P.leap"
    )
    os.system("rm -rf system_TIP3P.inpcrd")
    os.system("rm -rf system_TIP3P.parm7")
    os.system("rm -rf system_TIP3P.pdb")
    os.system("rm -rf system_TIP3P.inpcrd")
    os.system("rm -rf system_TIP3P.rst7")
    os.system("rm -rf system_TIP3P.prmtop")
    os.system("rm -rf system.pdb")
    os.system("rm -rf input_TIP3P.leap")
    os.system("rm -rf alanine-dipeptide-nowater.pdb")


def create_vectors(x):

    """
    Extracts peridic box information from the
    given line.

    """
    x = str(x)
    x = x.replace("Vec3", "")
    x = re.findall("\d*\.?\d+", x)
    for i in range(0, len(x)):
        x[i] = float(x[i])
    x = tuple(x)
    n = int(len(x) / 3)
    x = [x[i * n : (i + 1) * n] for i in range((len(x) + n - 1) // n)]
    return x


def simulated_annealing(
    parm="system_TIP3P.prmtop",
    rst="system_TIP3P.inpcrd",
    annealing_output_pdb="system_annealing_output.pdb",
    annealing_steps=100000,
    pdb_freq=100000,
    starting_temp=0,
    target_temp=300,
    temp_incr=3,
):

    """

    Performs simulated annealing of the system from
    0K to 300 K (default) using OpenMM MD engine and
    saves the last frame of the simulation to be
    accessed by the next simulation.

    Parameters
    ----------
    parm: str
        System's topology file

    rst: str
        System's coordinate file

    annealing_output_pdb: str
        System's output trajectory file

    annealing_steps: int
        Aneealing steps at each temperatrure jump

    pdb_freq: int
        Trajectory to be saved after every pdb_freq steps

    starting_temp: int
        Initial temperature of Simulated Annealing

    target_temp: int
        Final temperature of Simulated Annealing

    temp_incr: int
        Temmperature increase for every step

    """

    prmtop = AmberPrmtopFile(parm)
    inpcrd = AmberInpcrdFile(rst)
    annealing_system = prmtop.createSystem(
        nonbondedMethod=PME, nonbondedCutoff=1 * nanometer, constraints=HBonds
    )
    annealing_integrator = LangevinIntegrator(
        0 * kelvin, 1 / picosecond, 2 * femtoseconds
    )
    total_steps = ((target_temp / temp_incr) + 1) * annealing_steps
    annealing_temp_range = int((target_temp / temp_incr) + 1)
    annealing_platform = Platform.getPlatformByName("CUDA")
    annealing_properties = {"CudaDeviceIndex": "0", "CudaPrecision": "mixed"}
    annealing_simulation = Simulation(
        prmtop.topology,
        annealing_system,
        annealing_integrator,
        annealing_platform,
        annealing_properties,
    )
    annealing_simulation.context.setPositions(inpcrd.positions)
    if inpcrd.boxVectors is not None:
        annealing_simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
    annealing_simulation.minimizeEnergy()
    annealing_simulation.reporters.append(
        PDBReporter(annealing_output_pdb, pdb_freq)
    )
    simulated_annealing_last_frame = (
        annealing_output_pdb[:-4] + "_last_frame.pdb"
    )
    annealing_simulation.reporters.append(
        PDBReporter(simulated_annealing_last_frame, total_steps)
    )
    annealing_simulation.reporters.append(
        StateDataReporter(
            stdout,
            pdb_freq,
            step=True,
            time=True,
            potentialEnergy=True,
            totalSteps=total_steps,
            temperature=True,
            progress=True,
            remainingTime=True,
            speed=True,
            separator="\t",
        )
    )
    temp = starting_temp
    while temp <= target_temp:
        annealing_integrator.setTemperature(temp * kelvin)
        if temp == starting_temp:
            annealing_simulation.step(annealing_steps)
            annealing_simulation.saveState("annealing.state")
        else:
            annealing_simulation.loadState("annealing.state")
            annealing_simulation.step(annealing_steps)
        temp += temp_incr
    state = annealing_simulation.context.getState()
    print(state.getPeriodicBoxVectors())
    annealing_simulation_box_vectors = state.getPeriodicBoxVectors()
    print(annealing_simulation_box_vectors)
    with open("annealing_simulation_box_vectors.pkl", "wb") as f:
        pk.dump(annealing_simulation_box_vectors, f)
    print("Finshed NVT Simulated Annealing Simulation")


def npt_equilibration(
    parm="system_TIP3P.prmtop",
    npt_output_pdb="system_npt_output.pdb",
    pdb_freq=500000,
    npt_steps=5000000,
    target_temp=300,
    npt_pdb="system_annealing_output_last_frame.pdb",
):

    """

    Performs NPT equilibration MD of the system
    using OpenMM MD engine and saves the last
    frame of the simulation to be accessed by
    the next simulation.

    Parameters
    ----------
    parm: str
        System's topology file

    npt_output_pdb: str
        System's output trajectory file

    pdb_freq: int
        Trajectory to be saved after every pdb_freq steps

    npt_steps: int
        NPT simulation steps

    target_temp: int
        Temperature for MD simulation

    npt_pdb: str
        Last frame of the simulation

    """

    npt_init_pdb = PDBFile(npt_pdb)
    prmtop = AmberPrmtopFile(parm)
    npt_system = prmtop.createSystem(
        nonbondedMethod=PME, nonbondedCutoff=1 * nanometer, constraints=HBonds
    )
    barostat = MonteCarloBarostat(25.0 * bar, target_temp * kelvin, 25)
    npt_system.addForce(barostat)
    npt_integrator = LangevinIntegrator(
        target_temp * kelvin, 1 / picosecond, 2 * femtoseconds
    )
    npt_platform = Platform.getPlatformByName("CUDA")
    npt_properties = {"CudaDeviceIndex": "0", "CudaPrecision": "mixed"}
    npt_simulation = Simulation(
        prmtop.topology,
        npt_system,
        npt_integrator,
        npt_platform,
        npt_properties,
    )
    npt_simulation.context.setPositions(npt_init_pdb.positions)
    npt_simulation.context.setVelocitiesToTemperature(target_temp * kelvin)
    with open("annealing_simulation_box_vectors.pkl", "rb") as f:
        annealing_simulation_box_vectors = pk.load(f)
    annealing_simulation_box_vectors = create_vectors(
        annealing_simulation_box_vectors
    )
    npt_simulation.context.setPeriodicBoxVectors(
        annealing_simulation_box_vectors[0],
        annealing_simulation_box_vectors[1],
        annealing_simulation_box_vectors[2],
    )
    npt_last_frame = npt_output_pdb[:-4] + "_last_frame.pdb"
    npt_simulation.reporters.append(PDBReporter(npt_output_pdb, pdb_freq))
    npt_simulation.reporters.append(PDBReporter(npt_last_frame, npt_steps))
    npt_simulation.reporters.append(
        StateDataReporter(
            stdout,
            pdb_freq,
            step=True,
            time=True,
            potentialEnergy=True,
            totalSteps=npt_steps,
            temperature=True,
            progress=True,
            remainingTime=True,
            speed=True,
            separator="\t",
        )
    )
    npt_simulation.minimizeEnergy()
    npt_simulation.step(npt_steps)
    npt_simulation.saveState("npt_simulation.state")
    state = npt_simulation.context.getState()
    print(state.getPeriodicBoxVectors())
    npt_simulation_box_vectors = state.getPeriodicBoxVectors()
    print(npt_simulation_box_vectors)
    with open("npt_simulation_box_vectors.pkl", "wb") as f:
        pk.dump(npt_simulation_box_vectors, f)
    print("Finished NPT Simulation")


def nvt_equilibration(
    parm="system_TIP3P.prmtop",
    nvt_output_pdb="system_nvt_output.pdb",
    pdb_freq=500000,
    nvt_steps=5000000,
    target_temp=300,
    nvt_pdb="system_npt_output_last_frame.pdb",
):

    """

    Performs NVT equilibration MD of the system
    using OpenMM MD engine  saves the last
    frame of the simulation to be accessed by
    the next simulation.

    Parameters
    ----------
    parm: str
        System's topology file

    nvt_output_pdb: str
        System's output trajectory file

    pdb_freq: int
        Trajectory to be saved after every pdb_freq steps

    nvt_steps: int
        NVT simulation steps

    target_temp: int
        Temperature for MD simulation

    nvt_pdb: str
        Last frame of the simulation

    """

    nvt_init_pdb = PDBFile(nvt_pdb)
    prmtop = AmberPrmtopFile(parm)
    nvt_system = prmtop.createSystem(
        nonbondedMethod=PME, nonbondedCutoff=1 * nanometer, constraints=HBonds
    )
    nvt_integrator = LangevinIntegrator(
        target_temp * kelvin, 1 / picosecond, 2 * femtoseconds
    )
    nvt_platform = Platform.getPlatformByName("CUDA")
    nvt_properties = {"CudaDeviceIndex": "0", "CudaPrecision": "mixed"}
    nvt_simulation = Simulation(
        prmtop.topology,
        nvt_system,
        nvt_integrator,
        nvt_platform,
        nvt_properties,
    )
    nvt_simulation.context.setPositions(nvt_init_pdb.positions)
    nvt_simulation.context.setVelocitiesToTemperature(target_temp * kelvin)
    with open("npt_simulation_box_vectors.pkl", "rb") as f:
        npt_simulation_box_vectors = pk.load(f)
    npt_simulation_box_vectors = create_vectors(npt_simulation_box_vectors)
    nvt_simulation.context.setPeriodicBoxVectors(
        npt_simulation_box_vectors[0],
        npt_simulation_box_vectors[1],
        npt_simulation_box_vectors[2],
    )
    nvt_last_frame = nvt_output_pdb[:-4] + "_last_frame.pdb"
    nvt_simulation.reporters.append(PDBReporter(nvt_output_pdb, pdb_freq))
    nvt_simulation.reporters.append(PDBReporter(nvt_last_frame, nvt_steps))
    nvt_simulation.reporters.append(
        StateDataReporter(
            stdout,
            pdb_freq,
            step=True,
            time=True,
            potentialEnergy=True,
            totalSteps=nvt_steps,
            temperature=True,
            progress=True,
            remainingTime=True,
            speed=True,
            separator="\t",
        )
    )
    nvt_simulation.minimizeEnergy()
    nvt_simulation.step(nvt_steps)
    nvt_simulation.saveState("nvt_simulation.state")
    state = nvt_simulation.context.getState()
    print(state.getPeriodicBoxVectors())
    nvt_simulation_box_vectors = state.getPeriodicBoxVectors()
    print(nvt_simulation_box_vectors)
    with open("nvt_simulation_box_vectors.pkl", "wb") as f:
        pk.dump(nvt_simulation_box_vectors, f)
    print("Finished NVT Simulation")


def run_equilibration():

    """

    Runs systematic simulated annealing followed by
    NPT and NVT equilibration MD simulation.

    """

    cwd = os.getcwd()
    target_dir = cwd + "/" + "equilibration"
    os.system("rm -rf equilibration")
    os.system("mkdir equilibration")
    shutil.copy(
        cwd + "/" + "system_inputs" + "/" + "system_TIP3P.inpcrd",
        target_dir + "/" + "system_TIP3P.inpcrd",
    )
    shutil.copy(
        cwd + "/" + "system_inputs" + "/" + "system_TIP3P.parm7",
        target_dir + "/" + "system_TIP3P.parm7",
    )
    shutil.copy(
        cwd + "/" + "system_inputs" + "/" + "system_TIP3P.pdb",
        target_dir + "/" + "system_TIP3P.pdb",
    )
    shutil.copy(
        cwd + "/" + "system_inputs" + "/" + "system_TIP3P.prmtop",
        target_dir + "/" + "system_TIP3P.prmtop",
    )
    shutil.copy(
        cwd + "/" + "system_inputs" + "/" + "system_TIP3P.rst7",
        target_dir + "/" + "system_TIP3P.rst7",
    )
    shutil.copy(
        cwd + "/" + "system_inputs" + "/" + "system.pdb",
        target_dir + "/" + "system.pdb",
    )
    shutil.copy(
        cwd + "/" + "system_inputs" + "/" + "alanine-dipeptide-nowater.pdb",
        target_dir + "/" + "alanine-dipeptide-nowater.pdb",
    )
    shutil.copy(
        cwd + "/" + "system_inputs" + "/" + "input_TIP3P.leap",
        target_dir + "/" + "input_TIP3P.leap",
    )
    os.chdir(target_dir)
    simulated_annealing()
    npt_equilibration()
    nvt_equilibration()
    os.system("rm -rf system_TIP3P.inpcrd")
    os.system("rm -rf system_TIP3P.parm7")
    os.system("rm -rf system_TIP3P.pdb")
    os.system("rm -rf system_TIP3P.rst7")
    os.system("rm -rf system_TIP3P.prmtop")
    os.system("rm -rf system.pdb")
    os.system("rm -rf alanine-dipeptide-nowater.pdb")
    os.system("rm -rf input_TIP3P.leap")
    os.chdir(cwd)


def create_alanine_dipeptide_md_structures():

    cwd = os.getcwd()
    target_dir = cwd + "/" + "alanine_dipeptide_md"
    os.system("rm -rf alanine_dipeptide_md")
    os.system("mkdir alanine_dipeptide_md")
    shutil.copy(
        cwd + "/" + "equilibration" + "/" + "system_nvt_output_last_frame.pdb",
        target_dir + "/" + "system_nvt_output_last_frame.pdb",
    )
    os.chdir(target_dir)
    fix_cap_remove_nme("system_nvt_output_last_frame.pdb")
    fix_cap_replace_nme("system_nvt_output_last_frame.pdb")
    # Save the tleap script to file
    with open("final_input_TIP3P.leap", "w") as f:
        f.write(
            """
    source leaprc.protein.ff14SB
    source leaprc.water.tip3p
    set default FlexibleWater on
    set default PBRadii mbondi2
    pdb = loadpdb system_nvt_output_last_frame.pdb
    saveamberparm pdb system_final.prmtop system_final.inpcrd
    saveamberparm pdb system_final.parm7 system_final.rst7
    savepdb pdb system_final.pdb
    quit
    """
        )
    os.system("tleap -f final_input_TIP3P.leap")
    os.system("rm -rf leap.log")
    os.system("rm -rf system_nvt_output_last_frame.pdb")
    os.chdir(cwd)


def add_vec_inpcrd():

    """

    Adds box dimensions captured from the last saved
    frame of the NVT simulations to the inpcrd file.
    Only to be used when the box dimensions are not
    present in the inpcrd file.

    """

    cwd = os.getcwd()
    target_dir = cwd + "/" + "alanine_dipeptide_md"
    shutil.copy(
        cwd + "/" + "equilibration" + "/" + "nvt_simulation_box_vectors.pkl",
        target_dir + "/" + "nvt_simulation_box_vectors.pkl",
    )

    os.chdir(target_dir)
    with open("nvt_simulation_box_vectors.pkl", "rb") as f:
        nvt_simulation_box_vectors = pk.load(f)
    nvt_simulation_box_vectors = create_vectors(nvt_simulation_box_vectors)
    vectors = (
        (nvt_simulation_box_vectors[0][0]) * 10,
        (nvt_simulation_box_vectors[1][1]) * 10,
        (nvt_simulation_box_vectors[2][2]) * 10,
    )
    vectors = (
        round(vectors[0], 7),
        round(vectors[1], 7),
        round(vectors[2], 7),
    )
    last_line = (
        "  "
        + str(vectors[0])
        + "  "
        + str(vectors[1])
        + "  "
        + str(vectors[2])
        + "  90.0000000"
        + "  90.0000000"
        + "  90.0000000"
    )
    with open("system_final.inpcrd", "a+") as f:
        f.write(last_line)
    os.system("rm -rf nvt_simulation_box_vectors.pkl")
    os.chdir(cwd)


def add_vec_prmtop():

    """

    Adds box dimensions captured from the last saved
    frame of the NVT simulations to the prmtop file.
    Only to be used when the box dimensions are not
    present in the prmtop file.

    """

    cwd = os.getcwd()
    target_dir = cwd + "/" + "alanine_dipeptide_md"
    shutil.copy(
        cwd + "/" + "equilibration" + "/" + "nvt_simulation_box_vectors.pkl",
        target_dir + "/" + "nvt_simulation_box_vectors.pkl",
    )

    os.chdir(target_dir)
    with open("nvt_simulation_box_vectors.pkl", "rb") as f:
        nvt_simulation_box_vectors = pk.load(f)
    nvt_simulation_box_vectors = create_vectors(nvt_simulation_box_vectors)
    vectors = (
        nvt_simulation_box_vectors[0][0],
        nvt_simulation_box_vectors[1][1],
        nvt_simulation_box_vectors[2][2],
    )
    vectors = round(vectors[0], 7), round(vectors[1], 7), round(vectors[2], 7)
    oldbeta = "9.00000000E+01"
    x = str(vectors[0]) + str(0) + "E+" + "01"
    y = str(vectors[1]) + str(0) + "E+" + "01"
    z = str(vectors[2]) + str(0) + "E+" + "01"
    line1 = "%FLAG BOX_DIMENSIONS"
    line2 = "%FORMAT(5E16.8)"
    line3 = "  " + oldbeta + "  " + x + "  " + y + "  " + z
    with open("system_final.prmtop") as i, open(
        "system_intermediate_final.prmtop", "w"
    ) as f:
        for line in i:
            if line.startswith("%FLAG RADIUS_SET"):
                line = line1 + "\n" + line2 + "\n" + line3 + "\n" + line
            f.write(line)
    os.system("rm -rf system_final.prmtop")
    os.system("mv system_intermediate_final.prmtop system_final.prmtop")
    os.system("rm -rf nvt_simulation_box_vectors.pkl")
    os.chdir(cwd)




def explicit_md_input_alanine_dipeptide(imin = 0, irest = 0, ntx = 1, nstlim = 250000000, dt = 0.002, ntc = 2, 
                           ntf = 2, tol = 0.000001, iwrap = 1, ntb = 1, cut = 8.0, ntt = 3, 
                           temp0 = 300.0, gamma_ln = 1.0, ntpr = 500, ntwx = 500, ntwr = 500,
                           ntxo = 2, ioutfm = 1, ig = -1, ntwprt = 0, md_input_file = "md.in"):
    cwd = os.getcwd()
    target_dir = cwd + "/" + "alanine_dipeptide_md"
    os.chdir(target_dir)
    line_1 = "&cntrl"
    line_2 = "  " + "imin" + "=" + str(imin) + "," + "irest" + "=" + str(irest) + "," + "ntx" + "=" + str(ntx) + ","
    line_3 = "  " + "nstlim" + "=" + str(nstlim) + ", " + "dt" + "=" + str(dt) + "," + "ntc" + "=" + str(ntc) + ","
    line_4 = "  " + "ntf" + "=" + str(ntf) + "," + "tol" + "=" + str(tol) + "," + "iwrap" + "=" + str(iwrap) + ","
    line_5 = "  " + "ntb" + "=" + str(ntb) + "," + "cut" + "=" + str(cut) + "," + "ntt" + "=" + str(ntt) + ","
    line_6 = "  " + "temp0" + "=" + str(temp0) + "," + "gamma_ln" + "=" + str(gamma_ln) + "," + "ntpr" + "=" + str(ntpr) + ","
    line_7 = "  " + "ntwx" + "=" + str(ntwx) + "," + "ntwr" + "=" + str(ntwr) + "," + "ntxo" + "=" + str(ntxo) + ","
    line_8 = "  " + "ioutfm" + "=" + str(ioutfm) + "," + "ig" + "=" + str(ig) + "," + "ntwprt" + "=" + str(ntwprt) + ","
    line_9 = "&end"
    with open(md_input_file, "w") as f:
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
    os.chdir(cwd)


prepare_alanine_dipeptide()
run_equilibration()
create_alanine_dipeptide_md_structures()
add_vec_inpcrd()
add_vec_prmtop()
explicit_md_input_alanine_dipeptide()

