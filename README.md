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

from simtk.openmm.app import *
from numpy.linalg import norm
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import pickle as pk
import numpy as np
import re
################################################################################################################################################
target_temp = 300
annealing_steps = 10000
npt_steps = 10000000
nvt_steps = 10000000
pdb_freq_annealing = 10000
pdb_freq = 100000
################################################################################################################################################
def unit_vector(x):
    """Return a unit vector in the same direction as x."""
    y = np.array(x, dtype='float')
    return y / norm(y)

def cell_to_cellpar(cell, radians=False):
    """Returns the cell parameters [a, b, c, alpha, beta, gamma]. Angles are in degrees unless radian=True is used.
    """
    lengths = [np.linalg.norm(v) for v in cell]
    angles = []
    for i in range(3):
        j = i - 1
        k = i - 2
        ll = lengths[j] * lengths[k]
        if ll > 1e-16:
            x = np.dot(cell[j], cell[k]) / ll
            angle = 180.0 / np.pi * np.arccos(x)
        else:
            angle = 90.0
        angles.append(angle)
    if radians:
        angles = [angle * np.pi / 180 for angle in angles]
    return np.array(lengths + angles)

def cellpar_to_cell(cellpar, ab_normal=(0, 0, 1), a_direction=None):
    """Return a 3x3 cell matrix from cellpar=[a,b,c,alpha,beta,gamma]. Angles must be in degrees. The returned cell is orientated such that a and b are normal to `ab_normal` and a is parallel to the projection of `a_direction` in the a-b plane. Default `a_direction` is (1,0,0), unless this is parallel to `ab_normal`, in which case default `a_direction` is (0,0,1). The returned cell has the vectors va, vb and vc along the rows. The cell will be oriented such that va and vb are normal to `ab_normal` and va will be along the projection of `a_direction` onto the a-b plane.
    Example:
    >>> cell = cellpar_to_cell([1, 2, 4, 10, 20, 30], (0, 1, 1), (1, 2, 3))
    >>> np.round(cell, 3)
    array([[ 0.816, -0.408,  0.408],
           [ 1.992, -0.13 ,  0.13 ],
           [ 3.859, -0.745,  0.745]])
    """
    if a_direction is None:
        if np.linalg.norm(np.cross(ab_normal, (1, 0, 0))) < 1e-5:
            a_direction = (0, 0, 1)
        else:
            a_direction = (1, 0, 0)
    # Define rotated X,Y,Z-system, with Z along ab_normal and X along
    # the projection of a_direction onto the normal plane of Z.
    ad = np.array(a_direction)
    Z = unit_vector(ab_normal)
    X = unit_vector(ad - np.dot(ad, Z) * Z)
    Y = np.cross(Z, X)
    # Express va, vb and vc in the X,Y,Z-system
    alpha, beta, gamma = 90., 90., 90.
    if isinstance(cellpar, (int, float)):
        a = b = c = cellpar
    elif len(cellpar) == 1:
        a = b = c = cellpar[0]
    elif len(cellpar) == 3:
        a, b, c = cellpar
    else:
        a, b, c, alpha, beta, gamma = cellpar
    # Handle orthorhombic cells separately to avoid rounding errors
    eps = 2 * np.spacing(90.0, dtype=np.float64)  # around 1.4e-14
    # alpha
    if abs(abs(alpha) - 90) < eps:
        cos_alpha = 0.0
    else:
        cos_alpha = np.cos(alpha * np.pi / 180.0)
    # beta
    if abs(abs(beta) - 90) < eps:
        cos_beta = 0.0
    else:
        cos_beta = np.cos(beta * np.pi / 180.0)
    # gamma
    if abs(gamma - 90) < eps:
        cos_gamma = 0.0
        sin_gamma = 1.0
    elif abs(gamma + 90) < eps:
        cos_gamma = 0.0
        sin_gamma = -1.0
    else:
        cos_gamma = np.cos(gamma * np.pi / 180.0)
        sin_gamma = np.sin(gamma * np.pi / 180.0)
    # Build the cell vectors
    va = a * np.array([1, 0, 0])
    vb = b * np.array([cos_gamma, sin_gamma, 0])
    cx = cos_beta
    cy = (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    cz_sqr = 1. - cx * cx - cy * cy
    assert cz_sqr >= 0
    cz = np.sqrt(cz_sqr)
    vc = c * np.array([cx, cy, cz])
    # Convert to the Cartesian x,y,z-system
    abc = np.vstack((va, vb, vc))
    T = np.vstack((X, Y, Z))
    cell = np.dot(abc, T)
    return cell
################################################################################################################################################
#Simulated Annealing
parm = "system_TP4EW.parm7"
inpcrd = "system_TP4EW.inpcrd"
annealing_output_pdb = "system_annealing_output.pdb"
starting_temp = 0
temp_incr = 3
prmtop = AmberPrmtopFile(parm)
inpcrd = AmberInpcrdFile(inpcrd)
annealing_system = prmtop.createSystem(nonbondedMethod=PME,nonbondedCutoff=1*nanometer,constraints=HBonds)
annealing_integrator = LangevinIntegrator(0*kelvin, 1/picosecond, 2*femtoseconds)
total_steps = ((target_temp / temp_incr) + 1) * annealing_steps
annealing_temp_range = int((target_temp / temp_incr) + 1)
annealing_platform = Platform.getPlatformByName('CUDA')
annealing_properties = {'CudaDeviceIndex': '0', 'CudaPrecision': 'mixed'}
annealing_simulation = Simulation(prmtop.topology, annealing_system, annealing_integrator, annealing_platform,annealing_properties)
annealing_simulation.context.setPositions(inpcrd.positions)
annealing_simulation.minimizeEnergy()
annealing_simulation.reporters.append(PDBReporter(annealing_output_pdb, pdb_freq_annealing))
simulated_annealing_last_frame = annealing_output_pdb[:-4] + '_last_frame.pdb'
annealing_simulation.reporters.append(PDBReporter(simulated_annealing_last_frame, total_steps))
annealing_simulation.reporters.append(StateDataReporter(stdout,pdb_freq_annealing,step=True, time=True,potentialEnergy=True, totalSteps=total_steps,temperature=True,progress=True,remainingTime=True,speed=True, separator='\t'))
temp = starting_temp
while temp <= target_temp:
    annealing_integrator.setTemperature(temp*kelvin)
    if temp == starting_temp:
        annealing_simulation.step(annealing_steps)
        annealing_simulation.saveState('annealing.state')
    else:
        annealing_simulation.loadState('annealing.state')
        annealing_simulation.step(annealing_steps)
    temp += temp_incr
state = annealing_simulation.context.getState()
print(state.getPeriodicBoxVectors())
annealing_simulation_box_vectors = state.getPeriodicBoxVectors()
print(annealing_simulation_box_vectors)
with open('annealing_simulation_box_vectors.pkl', 'wb') as f:
    pk.dump(annealing_simulation_box_vectors, f)       
print("Finshed NVT Simulated Annealing Simulation")
################################################################################################################################################
#NPT Equilibration
parm = "system_TP4EW.parm7"
npt_output_pdb = "system_npt_output.pdb" 
npt_pdb = "system_annealing_output_last_frame.pdb"
npt_init_pdb = PDBFile(npt_pdb)
prmtop = AmberPrmtopFile(parm)
npt_system = prmtop.createSystem(nonbondedMethod=PME,nonbondedCutoff=1*nanometer,constraints=HBonds)
barostat = MonteCarloBarostat(25.0*bar,target_temp*kelvin, 25)
npt_system.addForce(barostat)
npt_integrator = LangevinIntegrator(target_temp*kelvin, 1/picosecond, 2*femtoseconds)
npt_platform = Platform.getPlatformByName('CUDA')
npt_properties = {'CudaDeviceIndex': '0', 'CudaPrecision': 'mixed'}
npt_simulation = Simulation(prmtop.topology, npt_system, npt_integrator, npt_platform, npt_properties)
npt_simulation.context.setPositions(npt_init_pdb.positions)
npt_simulation.context.setVelocitiesToTemperature(target_temp*kelvin)
npt_last_frame = npt_output_pdb[:-4] + '_last_frame.pdb' 
npt_simulation.reporters.append(PDBReporter(npt_output_pdb,pdb_freq))
npt_simulation.reporters.append(PDBReporter(npt_last_frame,npt_steps))
npt_simulation.reporters.append(StateDataReporter(stdout,pdb_freq,step=True, time=True, potentialEnergy=True,totalSteps=npt_steps,temperature=True,progress=True,  remainingTime=True,speed=True, separator='\t'))
npt_simulation.minimizeEnergy()
npt_simulation.step(npt_steps)
state = npt_simulation.context.getState()
print(state.getPeriodicBoxVectors())
npt_simulation_box_vectors = state.getPeriodicBoxVectors()
print(npt_simulation_box_vectors)
with open('npt_simulation_box_vectors.pkl', 'wb') as f:
    pk.dump(npt_simulation_box_vectors, f)       
print("Finished NPT Simulation")
################################################################################################################################################
pkl_file = 'npt_simulation_box_vectors.pkl'
with open(pkl_file, 'rb') as f:
    vectors = pk.load(f)
vector = re.findall("[+-]?\d+\.\d+", str(vectors))
vector = [float(i) for i in vector]
vector = [vector[0:3], vector[3:6], vector[6:]]
print(vector)
pdb_vectors = list(cell_to_cellpar(vector))
print(pdb_vectors)
edges = pdb_vectors [0:3]
edges = [i * 10 for i in edges]
angles = pdb_vectors [3:]
box_dim_1 = '{:<010}'.format(round(edges[0],6))
box_dim_2 = '{:<010}'.format(round(edges[1],6))
box_dim_3 = '{:<010}'.format(round(edges[2],6))
box_angle_1 = '{:<010}'.format(round(angles[0],6))
box_angle_2 = '{:<010}'.format(round(angles[1],6))
box_angle_3 = '{:<010}'.format(round(angles[2],6))
inpcrd_line = "  " + box_dim_1 + "  " + box_dim_2 + "  " + box_dim_3 + "  " + box_angle_1 + "  " + box_angle_2 + "  " + box_angle_3
print(inpcrd_line)
pdb_line = "CRYST1" + "   " + "%.3f" % round(edges[0], 3) + "   " + "%.3f" % round(edges[1], 3) + "   " + "%.3f" % round(edges[2], 3) + " " + "%.2f" % round(angles[0], 3) + " " + "%.2f" % round(angles[1], 3) + " " + "%.2f" % round(angles[2], 3) + " P 1           1"
print(pdb_line)
parm_vector_1 = "{:.8E}".format(edges[0])
parm_vector_2 = "{:.8E}".format(edges[1])
parm_vector_3 = "{:.8E}".format(edges[2])
parm_angle_1 = "{:.8E}".format(angles[0])
parm_angle_2 = "{:.8E}".format(angles[1])
parm_angle_3 = "{:.8E}".format(angles[2])
parm_line  = "  " + parm_angle_1 + "  " + parm_vector_1 + "  " + parm_vector_2 + "  " + parm_vector_3 
print(parm_line)

# Changing inpcrd dimensions 
inpcrd_file = 'system_TP4EW.inpcrd'
new_inpcrd_file = 'system_TP4EW_I.inpcrd'
with open(inpcrd_file) as f:
    inpcrd_lines = f.readlines()
print("inpcrd line to replace" , inpcrd_lines[-1])   
inpcrd_lines[-1] = inpcrd_line
with open(new_inpcrd_file, 'w') as f:
    for i in inpcrd_lines:
        f.write(i)
print("inpcrd line replaced with",inpcrd_line)

# Changing pdb dimensions 
pdb_file = 'system_npt_output_last_frame.pdb'
new_pdb_file = 'system_TP4EW_I.pdb'
with open(pdb_file) as f:
    pdb_lines = f.readlines()

lines_contain =[]
for i in range(len(pdb_lines)):
    if pdb_lines[i].startswith('CRYST1'):
        lines_contain.append(i)
print("pdb line to replace", pdb_lines[lines_contain[0]])   
         
pdb_lines[lines_contain[0]] = pdb_line + "\n"
with open(new_pdb_file, 'w') as f:
    for i in pdb_lines:
        f.write(i)
print("pdb line replaced with", pdb_line)
# Changing parm7 dimensions 
parm_file = 'system_TP4EW.parm7'
new_parm_file = 'system_TP4EW_I.parm7'
with open(parm_file) as f:
    parm_lines = f.readlines()
    
lines_contain =[]
for i in range(len(parm_lines)):
    if parm_lines[i].startswith('%FLAG BOX_DIMENSIONS '):
        lines_contain.append(i+2)
print("parm line to replace", parm_lines[lines_contain[0]])   
parm_lines[lines_contain[0]] = parm_line + "\n"
with open(new_parm_file, 'w') as f:
    for i in parm_lines:
        f.write(i)
print("parm line replaced with", parm_line)
################################################################################################################################################
#NVT Equilibration
parm = "system_TP4EW_I.parm7"
nvt_output_pdb = "system_nvt_output.pdb"
nvt_pdb = "system_TP4EW_I.pdb"       
nvt_init_pdb = PDBFile(nvt_pdb)
prmtop = AmberPrmtopFile(parm)
nvt_system = prmtop.createSystem(nonbondedMethod=PME,nonbondedCutoff=1*nanometer,constraints=HBonds)
nvt_integrator = LangevinIntegrator(target_temp*kelvin, 1/picosecond, 2*femtoseconds)
nvt_platform = Platform.getPlatformByName('CUDA')
nvt_properties = {'CudaDeviceIndex': '0', 'CudaPrecision': 'mixed'}
nvt_simulation = Simulation(prmtop.topology, nvt_system, nvt_integrator,nvt_platform,nvt_properties)
nvt_simulation.context.setPositions(nvt_init_pdb.positions)
nvt_simulation.context.setVelocitiesToTemperature(target_temp*kelvin)
nvt_last_frame = nvt_output_pdb[:-4] + '_last_frame.pdb'
nvt_simulation.reporters.append(PDBReporter(nvt_output_pdb,pdb_freq))
nvt_simulation.reporters.append(PDBReporter(nvt_last_frame,nvt_steps))
nvt_simulation.reporters.append(StateDataReporter(stdout,pdb_freq,step=True, time=True,potentialEnergy=True, totalSteps=nvt_steps,temperature=True,progress=True,remainingTime=True,speed=True, separator='\t'))
nvt_simulation.minimizeEnergy()
nvt_simulation.step(nvt_steps)
state = nvt_simulation.context.getState()
print(state.getPeriodicBoxVectors())
nvt_simulation_box_vectors = state.getPeriodicBoxVectors()
print(nvt_simulation_box_vectors)
with open('nvt_simulation_box_vectors.pkl', 'wb') as f:
    pk.dump(nvt_simulation_box_vectors, f)       
print("Finished NVT Simulation")
################################################################################################################################################
