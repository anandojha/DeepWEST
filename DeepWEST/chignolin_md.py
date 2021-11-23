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


def fix_cap_chignolin(pdb_file):

    """
    Removes the problematic H atom of the capped GLY residue.

    """

    remove_words = ["H   GLY A"]
    with open(pdb_file) as oldfile, open("intermediate.pdb", "w") as newfile:
        for line in oldfile:
            if not any(word in line for word in remove_words):
                newfile.write(line)
    command = "rm -rf " + pdb_file
    os.system(command)
    command = "mv intermediate.pdb " + pdb_file
    os.system(command)


def prepare_chignolin():
    """
    Prepares the chignolin system for Molecular Dynamics (MD) simulations. 
    Downloads the pdb structure from
    http://ambermd.org/tutorials/advanced/tutorial22/files/5PTI-DtoH-dry.pdb
    and parameterizes it using General Amber Force Field
    (GAFF).

    """
    os.system("curl -O https://files.rcsb.org/download/1UAO.pdb1.gz")
    os.system("gunzip 1UAO.pdb1.gz")
    os.system("mv 1UAO.pdb1 chignolin.pdb")
    os.system("rm -rf system_inputs")
    os.system("mkdir system_inputs")
    cwd = os.getcwd()
    target_dir = cwd + "/" + "system_inputs"
    os.system("pdb4amber -i chignolin.pdb -o system.pdb")
    # save the tleap script to file
    with open("input.leap", "w") as f:
        f.write(
            """
    source leaprc.protein.ff14SB
    pdb = loadpdb system.pdb
    charge pdb
    saveamberparm pdb system.prmtop system.inpcrd
    saveamberparm pdb system.parm7 system.rst7
    savepdb pdb system.pdb
    quit
    """
        )
    os.system("tleap -f input.leap")
    os.system("rm -rf leap.log")
    shutil.copy(
        cwd + "/" + "system.inpcrd", target_dir + "/" + "system.inpcrd"
    )
    shutil.copy(cwd + "/" + "system.parm7", target_dir + "/" + "system.parm7")
    shutil.copy(cwd + "/" + "system.pdb", target_dir + "/" + "system.pdb")
    shutil.copy(
        cwd + "/" + "system.prmtop", target_dir + "/" + "system.prmtop"
    )
    shutil.copy(cwd + "/" + "system.rst7", target_dir + "/" + "system.rst7")
    shutil.copy(cwd + "/" + "input.leap", target_dir + "/" + "input.leap")
    shutil.copy(
        cwd + "/" + "chignolin.pdb", target_dir + "/" + "chignolin.pdb"
    )
    os.system("rm -rf system_sslink")
    os.system("rm -rf system_nonprot.pdb")
    os.system("rm -rf system.pdb")
    os.system("rm -rf system_renum.txt")
    os.system("rm -rf system.inpcrd")
    os.system("rm -rf system.parm7")
    os.system("rm -rf system.rst7")
    os.system("rm -rf system.prmtop")
    os.system("rm -rf input.leap")
    os.system("rm -rf chignolin.pdb")


def simulated_annealing(
    parm="system.prmtop",
    rst="system.inpcrd",
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
    annealing_system = prmtop.createSystem(implicitSolvent=OBC2)
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


def nvt_equilibration(
    parm="system.prmtop",
    nvt_output_pdb="system_nvt_output.pdb",
    pdb_freq=500000,
    nvt_steps=5000000,
    target_temp=300,
    nvt_pdb="system_annealing_output_last_frame.pdb",
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
    nvt_system = prmtop.createSystem(implicitSolvent=OBC2)
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
    NVT equilibration MD simulation.

    """

    cwd = os.getcwd()
    target_dir = cwd + "/" + "equilibration"
    os.system("rm -rf equilibration")
    os.system("mkdir equilibration")
    shutil.copy(
        cwd + "/" + "system_inputs" + "/" + "system.inpcrd",
        target_dir + "/" + "system.inpcrd",
    )
    shutil.copy(
        cwd + "/" + "system_inputs" + "/" + "system.parm7",
        target_dir + "/" + "system.parm7",
    )
    shutil.copy(
        cwd + "/" + "system_inputs" + "/" + "system.pdb",
        target_dir + "/" + "system.pdb",
    )
    shutil.copy(
        cwd + "/" + "system_inputs" + "/" + "system.prmtop",
        target_dir + "/" + "system.prmtop",
    )
    shutil.copy(
        cwd + "/" + "system_inputs" + "/" + "system.rst7",
        target_dir + "/" + "system.rst7",
    )
    shutil.copy(
        cwd + "/" + "system_inputs" + "/" + "chignolin.pdb",
        target_dir + "/" + "chignolin.pdb",
    )
    shutil.copy(
        cwd + "/" + "system_inputs" + "/" + "input.leap",
        target_dir + "/" + "input.leap",
    )
    os.chdir(target_dir)
    simulated_annealing()
    nvt_equilibration()
    os.system("rm -rf system.inpcrd")
    os.system("rm -rf system.parm7")
    os.system("rm -rf system.pdb")
    os.system("rm -rf system.rst7")
    os.system("rm -rf system.prmtop")
    os.system("rm -rf chignolin.pdb")
    os.system("rm -rf input.leap")
    os.chdir(cwd)


def create_chignolin_md_structures():

    cwd = os.getcwd()
    target_dir = cwd + "/" + "chignolin_md"
    os.system("rm -rf chignolin_md")
    os.system("mkdir chignolin_md")
    shutil.copy(
        cwd + "/" + "equilibration" + "/" + "system_nvt_output_last_frame.pdb",
        target_dir + "/" + "system_nvt_output_last_frame.pdb",
    )
    os.chdir(target_dir)
    os.system(
        "pdb4amber -i system_nvt_output_last_frame.pdb -o intermediate_temp.pdb"
    )
    os.system("rm -rf intermediate_temp_renum.txt")
    os.system("rm -rf intermediate_temp_sslink")
    os.system("rm -rf intermediate_temp_nonprot.pdb")
    remove_words = ["H   GLY A"]
    with open("intermediate_temp.pdb") as oldfile, open(
        "intermediate.pdb", "w"
    ) as newfile:
        for line in oldfile:
            if not any(word in line for word in remove_words):
                newfile.write(line)
    # Save the tleap script to file
    with open("final_input.leap", "w") as f:
        f.write(
            """
    source leaprc.protein.ff14SB
    pdb = loadpdb intermediate.pdb
    saveamberparm pdb system_final.prmtop system_final.inpcrd
    saveamberparm pdb system_final.parm7 system_final.rst7
    savepdb pdb system_final.pdb
    quit
    """
        )
    os.system("tleap -f final_input.leap")
    os.system("rm -rf leap.log")
    os.system("rm -rf intermediate.pdb")
    os.system("rm -rf intermediate_temp.pdb")
    os.system("rm -rf system_nvt_output_last_frame.pdb")
    os.chdir(cwd)

def implicit_md_input_chignolin(imin = 0, irest = 0, ntx = 1, nstlim = 250000000, dt = 0.002, ntc = 2, 
                                tol = 0.000001, igb = 5,  cut = 1000.00, ntt = 3, temp0 = 300.0, 
                                gamma_ln = 1.0, ntpr = 500, ntwx = 500, ntwr = 500, ntxo = 2, ioutfm = 1, 
                                ig = -1, ntwprt = 0, md_input_file = "md.in"):
    cwd = os.getcwd()
    target_dir = cwd + "/" + "chignolin_md"
    os.chdir(target_dir)
    line_1 = "&cntrl"
    line_2 = "  " + "imin" + "=" + str(imin) + "," + "irest" + "=" + str(irest) + "," + "ntx" + "=" + str(ntx) + ","
    line_3 = "  " + "nstlim" + "=" + str(nstlim) + ", " + "dt" + "=" + str(dt) + "," + "ntc" + "=" + str(ntc) + ","
    line_4 = "  " + "tol" + "=" + str(tol) + "," + "igb" + "=" + str(igb) + "," + "cut" + "=" + str(cut) + ","
    line_5 = "  " + "ntt" + "=" + str(ntt) + "," + "temp0" + "=" + str(temp0) + "," + "gamma_ln" + "=" + str(gamma_ln) + ","
    line_6 = "  " + "ntpr" + "=" + str(ntpr) + "," + "ntwx" + "=" + str(ntwx) + "," + "ntwr" + "=" + str(ntwr) + ","
    line_7 = "  " + "ntxo" + "=" + str(ntxo) + "," + "ioutfm" + "=" + str(ioutfm) + "," + "ig" + "=" + str(ig) + ","
    line_8 = "  " + "ntwprt" + "=" + str(ntwprt) + "," 
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


prepare_chignolin()
run_equilibration()
create_chignolin_md_structures()
implicit_md_input_chignolin()

