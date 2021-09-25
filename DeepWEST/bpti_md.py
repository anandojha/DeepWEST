from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
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


def prepare_bpti():

    """

    Prepares the Bovine pancreatic trypsin inhibitor
    system for Molecular Dynamics (MD) simulations. The 
    function downloads the pdb structure from
    http://ambermd.org/tutorials/advanced/tutorial22/files/5PTI-DtoH-dry.pdb
    and parameterizes it using General Amber Force Field
    (GAFF).

    """

    os.system("curl -O http://ambermd.org/tutorials/advanced/tutorial22/files/5PTI-DtoH-dry.pdb")
    os.system("rm -rf system_inputs_bpti")
    # Removes any existing directory named system_inputs_bpti
    os.system("mkdir system_inputs_bpti")
    # Creates a directory named system_inputs_bpti
    cwd = os.getcwd()
    target_dir = cwd + "/" + "system_inputs_bpti"
    # save the tleap script to file
    with open("input_TIP4P.leap", "w") as f:
        f.write(
            """
    source leaprc.protein.ff14SB
    loadOff solvents.lib
    loadOff tip4pbox.off
    loadOff tip4pewbox.off
    source leaprc.water.tip4pew
    HOH = TP4
    pdb = loadpdb 5PTI-DtoH-dry.pdb
    bond pdb.55.SG pdb.5.SG
    bond pdb.30.SG pdb.51.SG
    bond pdb.14.SG pdb.38.SG
    charge pdb
    addions2 pdb Cl- 6
    charge pdb
    solvatebox pdb TIP4PEWBOX 12.0
    saveamberparm pdb system_TIP4P.prmtop system_TIP4P.inpcrd
    saveamberparm pdb system_TIP4P.parm7 system_TIP4P.rst7
    savepdb pdb system_TIP4P.pdb
    quit
    """
        )
    os.system("tleap -f input_TIP4P.leap")
    os.system("rm -rf leap.log")
    shutil.copy(cwd + "/" + "system_TIP4P.inpcrd", target_dir + "/" + "system_TIP4P.inpcrd")
    shutil.copy(cwd + "/" + "system_TIP4P.parm7", target_dir + "/" + "system_TIP4P.parm7")
    shutil.copy(cwd + "/" + "system_TIP4P.pdb", target_dir + "/" + "system_TIP4P.pdb")
    shutil.copy(cwd + "/" + "system_TIP4P.prmtop", target_dir + "/" + "system_TIP4P.prmtop")
    shutil.copy(cwd + "/" + "system_TIP4P.rst7", target_dir + "/" + "system_TIP4P.rst7")
    shutil.copy(cwd + "/" + "input_TIP4P.leap", target_dir + "/" + "input_TIP4P.leap")
    shutil.copy(cwd + "/" + "5PTI-DtoH-dry.pdb", target_dir + "/" + "5PTI-DtoH-dry.pdb")
    os.system("rm -rf system_TIP4P.inpcrd")
    os.system("rm -rf system_TIP4P.parm7")
    os.system("rm -rf system_TIP4P.pdb")
    os.system("rm -rf system_TIP4P.rst7")
    os.system("rm -rf system_TIP4P.prmtop")
    os.system("rm -rf input_TIP4P.leap")
    os.system("rm -rf 5PTI-DtoH-dry.pdb")


def simulated_annealing(
    parm="system_TIP4P.prmtop",
    rst="system_TIP4P.inpcrd",
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
    annealing_system = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=1 * nanometer, constraints=HBonds)
    annealing_integrator = LangevinIntegrator(0 * kelvin, 1 / picosecond, 2 * femtoseconds)
    total_steps = ((target_temp / temp_incr) + 1) * annealing_steps
    annealing_temp_range = int((target_temp / temp_incr) + 1)
    annealing_platform = Platform.getPlatformByName("CUDA")
    annealing_properties = {"CudaDeviceIndex": "0", "CudaPrecision": "mixed"}
    annealing_simulation = Simulation(prmtop.topology, annealing_system, annealing_integrator, annealing_platform, annealing_properties)
    annealing_simulation.context.setPositions(inpcrd.positions)
    if inpcrd.boxVectors is not None:
        annealing_simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
    annealing_simulation.minimizeEnergy()
    annealing_simulation.reporters.append(PDBReporter(annealing_output_pdb, pdb_freq))
    simulated_annealing_last_frame = (annealing_output_pdb[:-4] + "_last_frame.pdb")
    annealing_simulation.reporters.append(PDBReporter(simulated_annealing_last_frame, total_steps))
    annealing_simulation.reporters.append(StateDataReporter(stdout, pdb_freq, step=True, time=True, potentialEnergy=True, totalSteps=total_steps, temperature=True, progress=True, remainingTime=True, speed=True, separator="\t"))
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


def npt_equilibration_bpti(
    parm="system_TIP4P.prmtop",
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
        nonbondedMethod=PME, nonbondedCutoff=1 * nanometer, constraints=HBonds)
    barostat = MonteCarloBarostat(25.0 * bar, target_temp * kelvin, 25)
    npt_system.addForce(barostat)
    npt_integrator = LangevinIntegrator(target_temp * kelvin, 1 / picosecond, 2 * femtoseconds)
    npt_platform = Platform.getPlatformByName("CUDA")
    npt_properties = {"CudaDeviceIndex": "0", "CudaPrecision": "mixed"}
    npt_simulation = Simulation(prmtop.topology, npt_system, npt_integrator, npt_platform, npt_properties)
    npt_simulation.context.setPositions(npt_init_pdb.positions)
    npt_simulation.context.setVelocitiesToTemperature(target_temp * kelvin)
    with open("annealing_simulation_box_vectors.pkl", "rb") as f:
        annealing_simulation_box_vectors = pk.load(f)
    annealing_simulation_box_vectors = create_vectors(annealing_simulation_box_vectors)
    npt_simulation.context.setPeriodicBoxVectors(annealing_simulation_box_vectors[0], annealing_simulation_box_vectors[1], annealing_simulation_box_vectors[2])
    npt_last_frame = npt_output_pdb[:-4] + "_last_frame.pdb"
    npt_simulation.reporters.append(PDBReporter(npt_output_pdb, pdb_freq))
    npt_simulation.reporters.append(PDBReporter(npt_last_frame, npt_steps))
    npt_simulation.reporters.append(StateDataReporter(stdout, pdb_freq, step=True, time=True, potentialEnergy=True, totalSteps=npt_steps, temperature=True, progress=True, remainingTime=True, speed=True, separator="\t"))
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


def nvt_equilibration_bpti(
    parm="system_TIP4P.prmtop",
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
        nonbondedMethod=PME, nonbondedCutoff=1 * nanometer, constraints=HBonds)
    nvt_integrator = LangevinIntegrator(target_temp * kelvin, 1 / picosecond, 2 * femtoseconds)
    nvt_platform = Platform.getPlatformByName("CUDA")
    nvt_properties = {"CudaDeviceIndex": "0", "CudaPrecision": "mixed"}
    nvt_simulation = Simulation(prmtop.topology, nvt_system, nvt_integrator, nvt_platform, nvt_properties)
    nvt_simulation.context.setPositions(nvt_init_pdb.positions)
    nvt_simulation.context.setVelocitiesToTemperature(target_temp * kelvin)
    with open("npt_simulation_box_vectors.pkl", "rb") as f:
        npt_simulation_box_vectors = pk.load(f)
    npt_simulation_box_vectors = create_vectors(npt_simulation_box_vectors)
    nvt_simulation.context.setPeriodicBoxVectors(npt_simulation_box_vectors[0], npt_simulation_box_vectors[1], npt_simulation_box_vectors[2])
    nvt_last_frame = nvt_output_pdb[:-4] + "_last_frame.pdb"
    nvt_simulation.reporters.append(PDBReporter(nvt_output_pdb, pdb_freq))
    nvt_simulation.reporters.append(PDBReporter(nvt_last_frame, nvt_steps))
    nvt_simulation.reporters.append(StateDataReporter(stdout, pdb_freq, step=True, time=True, potentialEnergy=True, totalSteps=nvt_steps, temperature=True, progress=True, remainingTime=True, speed=True, separator="\t"))
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


def run_equilibration_bpti():

    """

    Runs systematic simulated annealing followed by
    NPT and NVT equilibration MD simulation.

    """

    cwd = os.getcwd()
    target_dir = cwd + "/" + "equilibration_bpti"
    os.system("rm -rf equilibration_bpti")
    os.system("mkdir equilibration_bpti")
    shutil.copy(cwd + "/" + "system_inputs_bpti" + "/" + "system_TIP4P.inpcrd", target_dir + "/" + "system_TIP4P.inpcrd")
    shutil.copy(cwd + "/" + "system_inputs_bpti" + "/" + "system_TIP4P.parm7", target_dir + "/" + "system_TIP4P.parm7")
    shutil.copy(cwd + "/" + "system_inputs_bpti" + "/" + "system_TIP4P.pdb", target_dir + "/" + "system_TIP4P.pdb")
    shutil.copy(cwd + "/" + "system_inputs_bpti" + "/" + "system_TIP4P.prmtop", target_dir + "/" + "system_TIP4P.prmtop")
    shutil.copy(cwd + "/" + "system_inputs_bpti" + "/" + "system_TIP4P.rst7", target_dir + "/" + "system_TIP4P.rst7")
    shutil.copy(cwd + "/" + "system_inputs_bpti" + "/" + "5PTI-DtoH-dry.pdb", target_dir + "/" + "5PTI-DtoH-dry.pdb")
    shutil.copy(cwd + "/" + "system_inputs_bpti" + "/" + "input_TIP4P.leap", target_dir + "/" + "input_TIP4P.leap")
    os.chdir(target_dir)
    simulated_annealing()
    npt_equilibration_bpti()
    nvt_equilibration_bpti()
    os.system("rm -rf system_TIP4P.inpcrd")
    os.system("rm -rf system_TIP4P.parm7")
    os.system("rm -rf system_TIP4P.pdb")
    os.system("rm -rf system_TIP4P.rst7")
    os.system("rm -rf system_TIP4P.prmtop")
    os.system("rm -rf 5PTI-DtoH-dry.pdb")
    os.system("rm -rf input_TIP4P.leap")
    os.chdir(cwd)


def create_bpti_md():

    """
    Prepares starting structures for Amber MD simulations.
    All input files required to run Amber MD simulations are
    placed in the bpti_md directory.

    """

    cwd = os.getcwd()
    target_dir = cwd + "/" + "bpti_md"
    os.system("rm -rf bpti_md")
    os.system("mkdir bpti_md")
    shutil.copy(cwd + "/" + "equilibration_bpti" + "/" + "system_nvt_output_last_frame.pdb", target_dir + "/" + "system_nvt_output_last_frame.pdb")
    os.chdir(target_dir)
    os.system("pdb4amber -i system_nvt_output_last_frame.pdb -o intermediate_temp.pdb")
    os.system("rm -rf intermediate_temp_renum.txt")
    os.system("rm -rf intermediate_temp_sslink")
    os.system("rm -rf intermediate_temp_nonprot.pdb")
    remove_words = ["H   ARG A   1"]
    with open("intermediate_temp.pdb") as oldfile, open("intermediate.pdb", "w") as newfile:
        for line in oldfile:
            if not any(word in line for word in remove_words):
                newfile.write(line)
    # Save the tleap script to file
    with open("final_input_TIP4P.leap", "w") as f:
        f.write(
            """
    source leaprc.protein.ff14SB
    source leaprc.water.tip4pew
    pdb = loadpdb intermediate.pdb
    charge pdb
    saveamberparm pdb system_final.prmtop system_final.inpcrd
    saveamberparm pdb system_final.parm7 system_final.rst7
    savepdb pdb system_final.pdb
    quit
    """
        )
    os.system("tleap -f final_input_TIP4P.leap")
    os.system("rm -rf leap.log")
    os.system("rm -rf leap.log")
    os.system("rm -rf intermediate.pdb")
    os.system("rm -rf intermediate_temp.pdb")
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
    target_dir = cwd + "/" + "bpti_md"
    shutil.copy(cwd + "/" + "equilibration_bpti" + "/" + "nvt_simulation_box_vectors.pkl", target_dir + "/" + "nvt_simulation_box_vectors.pkl")
    os.chdir(target_dir)
    with open("nvt_simulation_box_vectors.pkl", "rb") as f:
        nvt_simulation_box_vectors = pk.load(f)
    nvt_simulation_box_vectors = create_vectors(nvt_simulation_box_vectors)
    vectors = ((nvt_simulation_box_vectors[0][0]) * 10, (nvt_simulation_box_vectors[1][1]) * 10, (nvt_simulation_box_vectors[2][2]) * 10)
    vectors = (round(vectors[0], 7), round(vectors[1], 7), round(vectors[2], 7))
    last_line = ("  " + str(vectors[0]) + "  " + str(vectors[1]) + "  " + str(vectors[2]) + "  90.0000000" + "  90.0000000" + "  90.0000000")
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
    target_dir = cwd + "/" + "bpti_md"
    shutil.copy(cwd + "/" + "equilibration_bpti" + "/" + "nvt_simulation_box_vectors.pkl", target_dir + "/" + "nvt_simulation_box_vectors.pkl")
    os.chdir(target_dir)
    with open("nvt_simulation_box_vectors.pkl", "rb") as f:
        nvt_simulation_box_vectors = pk.load(f)
    nvt_simulation_box_vectors = create_vectors(nvt_simulation_box_vectors)
    vectors = (nvt_simulation_box_vectors[0][0], nvt_simulation_box_vectors[1][1], nvt_simulation_box_vectors[2][2])
    vectors = round(vectors[0], 7), round(vectors[1], 7), round(vectors[2], 7)
    oldbeta = "9.00000000E+01"
    x = str(vectors[0]) + str(0) + "E+" + "01"
    y = str(vectors[1]) + str(0) + "E+" + "01"
    z = str(vectors[2]) + str(0) + "E+" + "01"
    line1 = "%FLAG BOX_DIMENSIONS"
    line2 = "%FORMAT(5E16.8)"
    line3 = "  " + oldbeta + "  " + x + "  " + y + "  " + z
    with open("system_final.prmtop") as i, open("system_intermediate_final.prmtop", "w") as f:
        for line in i:
            if line.startswith("%FLAG RADIUS_SET"):
                line = line1 + "\n" + line2 + "\n" + line3 + "\n" + line
            f.write(line)
    os.system("rm -rf system_final.prmtop")
    os.system("mv system_intermediate_final.prmtop system_final.prmtop")
    os.system("rm -rf nvt_simulation_box_vectors.pkl")
    os.chdir(cwd)

def explicit_md_input_bpti(imin = 0, irest = 0, ntx = 1, nstlim = 250000000, dt = 0.002, ntc = 2, 
                           ntf = 2, tol = 0.000001, iwrap = 1, ntb = 1, cut = 8.0, ntt = 3, 
                           temp0 = 300.0, gamma_ln = 1.0, ntpr = 500, ntwx = 500, ntwr = 500,
                           ntxo = 2, ioutfm = 1, ig = -1, ntwprt = 0, md_input_file = "md.in"):
    cwd = os.getcwd()
    target_dir = cwd + "/" + "bpti_md"
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

def run_bpti_md_cpu(md_input_file = "md.in", output_file = "system_final.out",
                    prmtopfile = "system_final.prmtop", inpcrd_file = "system_final.inpcrd", 
                    rst_file = "system_final.rst", traj_file = "system_final.nc"):
    cwd = os.getcwd()
    target_dir = cwd + "/" + "bpti_md"
    os.chdir(target_dir)
    command = "sander -O -i " + md_input_file + " -o " + output_file + " -p " + prmtopfile + " -c " + inpcrd_file + " -r " + rst_file + " -x " +  traj_file
    print("Running Amber MD simulations")
    os.system(command)
    print("Finished Amber MD simulations")
    os.chdir(cwd)
    
prepare_bpti()
run_equilibration_bpti()
create_bpti_md()
add_vec_inpcrd()
add_vec_prmtop()
explicit_md_input_bpti()
#run_bpti_md_cpu()
