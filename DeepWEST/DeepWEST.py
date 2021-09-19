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
import mdtraj as md
import numpy as np
import itertools
import fnmatch
import shutil
import scipy
import os
import re

################ Common Functions ################

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

################ Common Functions ################

################ Chignolin Functions ################

def create_chignolin_md_inputs(url = "https://files.rcsb.org/download/1UAO.pdb1.gz", 
                               ref_pdb = "chignolin.pdb"):
    command = "curl -O " + url
    os.system(command)
    command = "gunzip 1UAO.pdb1.gz"
    os.system(command)
    command = "mv 1UAO.pdb1 " + ref_pdb
    os.system(command)
    line_1 = "source leaprc.protein.ff14SB"
    line_2 = "pdb = loadpdb " + ref_pdb
    line_3 = "saveamberparm pdb " + ref_pdb[:-4] + ".prmtop " + ref_pdb[:-4] + ".inpcrd"
    line_4 = "savepdb pdb " + ref_pdb[:-4] + "_for_amber.pdb"
    line_5 = "quit"
    with open("chig.leap", "w") as f:
        f.write("    " + "\n")
        f.write(line_1 + "\n")
        f.write(line_2 + "\n")
        f.write(line_3 + "\n")
        f.write(line_4 + "\n")
        f.write(line_5 + "\n")
    command = "tleap -f chig.leap"
    os.system(command)
    command = "rm -rf chig.leap leap.log"
    os.system(command)
    
def input_chig_implicit_md(imin = 0, irest = 0, ntx = 1, dt = 0.002, ntc = 2, tol = 0.000001, 
                         igb = 5, cut = 1000.00, ntt = 3, temp0 = 300.0, gamma_ln = 1.0, 
                         ntpr = 500, ntwr = 500, ntxo = 2, ioutfm = 1, ig = -1, ntwprt = 0, 
                         md_input_file = "md.in", nstlim = 2500000000, ntwx = 25000):
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
        
def run_chignolin_md_cpu(md_input_file = "md.in", output_file = "chignolin.out",
                         prmtopfile = "chignolin.prmtop", inpcrd_file = "chignolin.inpcrd", 
                         rst_file = "chignolin.rst", traj_file = "chignolin.nc"):

    command = "sander -O -i " + md_input_file + " -o " + output_file + " -p " + prmtopfile + " -c " + inpcrd_file + " -r " + rst_file + " -x " +  traj_file
    print("Running Amber MD simulations")
    os.system(command)
    
def create_chignolin_traj_for_molearn(traj="chignolin.nc", ref_pdb="chignolin_for_amber.pdb", 
                                      start=0, stop=100000, stride=1, traj_pdb="chig_multi.pdb"):
    topology = md.load(ref_pdb).topology
    print(topology)
    trajec = md.load(traj, top=ref_pdb)
    print(trajec)
    trajec = trajec[start:stop:stride]
    print(trajec)
    trajec.save_pdb(traj_pdb, force_overwrite=True)  

"""
create_chignolin_md_inputs()
input_chig_implicit_md()
run_chignolin_md_cpu()
create_chignolin_traj_for_molearn()
"""
################ Chignolin Functions ################

################ Alanine Dipeptide Functions ################

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

def fix_cap_replace_nme_H(pdb_file):

    """
    Replaces the hydrogen atoms of the
    capped NME residue with a standard name.

    """

    fin = open(pdb_file, "rt")
    data = fin.read()
    data = data.replace("H1  NME", "H31 NME")
    data = data.replace("H2  NME", "H32 NME")
    data = data.replace("H3  NME", "H33 NME")
    fin.close()
    fin = open(pdb_file, "wt")
    fin.write(data)
    fin.close()

def create_alanine_dipeptide_md_inputs(url = "http://ftp.imp.fu-berlin.de/pub/cmb-data/alanine-dipeptide-nowater.pdb", 
                               ref_pdb = "alanine_dipeptide.pdb"):
    command = "curl -O " + url
    os.system(command)
    command = "mv alanine-dipeptide-nowater.pdb " + ref_pdb
    os.system(command)
    file1 = open(ref_pdb, "r")
    file2 = open("intermediate.pdb", "w")
    for line in file1.readlines():
        if "HH" not in line:
            file2.write(line) 
    file1.close()
    file2.close()
    command = "mv intermediate.pdb " + ref_pdb 
    os.system(command)
    line_1 = "source leaprc.protein.ff14SB"
    line_2 = "pdb = loadpdb " + ref_pdb
    line_3 = "saveamberparm pdb " + ref_pdb[:-4] + ".prmtop " + ref_pdb[:-4] + ".inpcrd"
    line_4 = "savepdb pdb " + ref_pdb[:-4] + "_for_amber.pdb"
    line_5 = "quit"
    with open("alad.leap", "w") as f:
        f.write("    " + "\n")
        f.write(line_1 + "\n")
        f.write(line_2 + "\n")
        f.write(line_3 + "\n")
        f.write(line_4 + "\n")
        f.write(line_5 + "\n")
    command = "tleap -f alad.leap"
    os.system(command)
    command = "rm -rf alad.leap leap.log"
    os.system(command)
    
def save_alanine_dipeptide_solv_to_no_solvent(traj="alanine_dipeptide.nc", 
                                              start=0, stop=100000, stride=1, 
                                              traj_pdb="alanine_dipeptide_.nc",
                                              top = "alanine_dipeptide.prmtop"):
    trajec = md.load(traj, top=top)
    print(trajec)
    trajec = trajec.remove_solvent()
    trajec = trajec[start:stop:stride]
    print(trajec)
    trajec.save_netcdf(traj_pdb, force_overwrite=True)  
    
def create_alanine_dipeptide_traj_for_molearn(traj="alanine_dipeptide_.nc", 
                                              ref_pdb="alanine_dipeptide_for_amber.pdb", 
                                              start=0, stop=100000, stride=1, 
                                              traj_pdb="alad_multi.pdb"):
    topology = md.load(ref_pdb).topology
    print(topology)
    trajec = md.load(traj, top=ref_pdb)
    print(trajec)
    trajec = trajec[start:stop:stride]
    print(trajec)
    trajec.save_pdb(traj_pdb, force_overwrite=True)  

"""
create_alanine_dipeptide_md_inputs()
save_alanine_dipeptide_solv_to_no_solvent()
create_alanine_dipeptide_traj_for_molearn()
"""
################ Alanine Dipeptide Functions ################

################ BPTI Functions ################

def create_bpti_traj_for_molearn(traj = "bpti.nc", top = "bpti.prmtop", traj_pdb = "bpti_multi.pdb", 
                                 start = 0, stop = 100000, stride = 1):

    trajec = md.load(traj, top=top)
    print(trajec)
    trajec = trajec.remove_solvent()
    trajec = trajec[start:stop:stride]
    print(trajec)
    trajec.save_pdb(traj_pdb, force_overwrite=True)

################ BPTI Functions ################

################ K-Means Clustering #############

def tsne_visualize(traj_data):
    tsne = TSNE(n_components=2, perplexity=100)
    tsne_dims = tsne.fit_transform(traj_data)
    sns.scatterplot(x=tsne_dims[:, 0], y=tsne_dims[:, 1], alpha=0.5)
    plt.title("2-D TSNE")
    plt.show()
    return tsne_dims


def experiment_with_k_means(traj_data, tsne_dims):
    k_range = np.arange(2, 9)
    # training k-means for 7 different values of num_clusters or k (kernels)
    trials_per_k = 5
    total_k = k_range.shape[0]
    score = np.zeros(total_k)
    for k_num in range(0, total_k):
        kmeans = KMeans(n_clusters=k_range[k_num])
        trial_score = np.zeros(trials_per_k)
        for trial in range(0, trials_per_k):
            labels_kmeans = kmeans.fit_predict(traj_data)
            trial_score[trial] = metrics.davies_bouldin_score(
                traj_data, labels_kmeans
            )
            # For clustering Davies Bouldin score or silhouette score is a good metric
        score[k_num] = np.median(trial_score)
        # Visualising the clusters using tsne fitted dimensions from the previous block
        sns.scatterplot(
            x=tsne_dims[:, 0],
            y=tsne_dims[:, 1],
            hue=labels_kmeans,
            palette="bright",
            alpha=0.5,
        )
        plt.title("2-D TSNE for K-Means with k = " + str(k_range[k_num]))
        plt.show()
        print("For %d clusters, score is %f " % (k_num + 2, score[k_num]))
    max_score = np.max(score)
    best_num_clusters = np.where(score == max_score)
    plt.plot(k_range, score)
    plt.title("Clustering score")
    plt.show()
    return best_num_clusters


def gmm(traj_data, best_num_clusters):
    num_clusters = best_num_clusters[0][0] + 2
    gm = GaussianMixture(n_components=num_clusters, random_state=0).fit(
        traj_data
    )
    posterior_probabs = gm.predict_proba(traj_data)
    print(posterior_probabs.shape)
    return posterior_probabs


def get_clustered_indices(traj_data, posterior_probabs, num_traj_indices):

    """
    Returns the indices of the trajectory points that are highly
    probable to belong to the smallest cluster (slowest state)

    """
    labels_final = np.argmax(posterior_probabs, axis=1)
    num_trajs = np.bincount(labels_final)
    slow_state_idx = np.argmin(num_trajs)
    num_clusters = posterior_probabs.shape[1]
    traj_idcs = [
        np.where(
            labels_final
            == np.multiply(np.ones_like(labels_final), slow_state_idx)
        )
    ]
    probs = np.zeros((len(traj_idcs[0][0])))
    for i, traj in enumerate(traj_idcs[0][0]):
        probs[i] = posterior_probabs[traj, slow_state_idx]
    probs_sorted = np.argsort(probs)
    return probs_sorted[:num_traj_indices], labels_final


def plot_RC(dihedral_data, labels):
    fig, ax = plt.subplots()
    for i in range(num_clusters):
        coor_train = np.where(labels == i)[0]
        ax.scatter(
            dihedral[coor_train, 0], dihedral[coor_train, 1], s=5, label=i
        )
    ax.legend()
    plt.axes = [[-np.pi, np.pi], [-np.pi, np.pi]]
    plt.show()


def print_states_pie_chart():
    coors = []
    maxi = np.max(pred_ord, axis=1)
    for i in range(output_size):
        coors.append(len(np.where(pred_ord[:, i] == maxi)[0]))
    fig1, ax1 = plt.subplots()
    ax1.pie(np.array(coors), autopct="%1.2f%%", startangle=90)
    ax1.axis("equal")
    # Equal aspect ratio ensures that pie is drawn as a circle.
    print("States population: " + str(np.array(coors) / len(maxi) * 100) + "%")
    plt.show()


def pdbs_from_indices(indices, traj_file, ref_pdb):
    os.system("rm -rf westpa_dir")
    os.system("mkdir westpa_dir")
    for i in indices:
        traj_frame = md.load_frame(filename=traj_file, top=ref_pdb, index=i)
        pdb_name = str(i) + ".pdb"
        pdb_path = os.path.join(os.getcwd(), "westpa_dir/" + pdb_name)
        traj_frame.save_pdb(pdb_path, force_overwrite=True)

################ K-Means Clustering #############

################ VAMPnet ################

class VampnetTools(object):

    """
    Wrapper for functions in VAMPnet.

    Parameters
    ----------

    epsilon: float, optional, default = 1e-10
        threshold for eigenvalues to be considered different
        from zero, used to prevent ill-conditioning problems
        during inversion of auto-covariance matrices.

    k_eig: int, optional, default = 0
        number of eigenvalues, or singular values, to be
        considered while calculating the VAMP score. If
        k_eig is higher than zero, only the top k_eig
        values will be considered, otherwise the
        algorithms will use all the available singular
        eigenvalues.

    """

    def __init__(self, epsilon=1e-10, k_eig=0):
        self._epsilon = epsilon
        self._k_eig = k_eig

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = value

    @property
    def k_eig(self):
        return self._k_eig

    @k_eig.setter
    def k_eig(self, value):
        self._k_eig = value

    def loss_VAMP(self, y_true, y_pred):

        """
        Calculates gradient of the VAMP-1 score calculated
        with respect to the network lobes. Shrinkage algorithm
        guarantees that auto-covariance matrices are positive
        and inverse square root exists.


        Parameters
        ----------
        y_true: tensorflow tensor
            parameter not needed for the calculation,
            added to comply with Keras rules for
            loss functions format.

        y_pred: tensorflow tensor with shape
                [batch_size, 2 * output_size]
            output of the two lobes of the
            network

        Returns
        -------
        loss_score: tensorflow tensor with shape
                   [batch_size, 2 * output_size]
            gradient of the VAMP-1 score

        """
        # Remove the mean from the data
        x, y, batch_size, output_size = self._prep_data(y_pred)

        # Calculate the inverse root of the auto-covariance matrices, and the
        # cross-covariance matrix
        matrices = self._build_vamp_matrices(x, y, batch_size)
        cov_00_ir, cov_11_ir, cov_01 = matrices

        vamp_matrix = tf.matmul(cov_00_ir, tf.matmul(cov_01, cov_11_ir))
        D, U, V = tf.linalg.svd(vamp_matrix, full_matrices=True)
        diag = tf.linalg.diag(D)

        # Base-changed covariance matrices
        x_base = tf.matmul(cov_00_ir, U)
        y_base = tf.matmul(cov_11_ir, V)

        # Calculate the gradients
        nabla_01 = tf.matmul(x_base, y_base, transpose_b=True)
        nabla_00 = -0.5 * tf.matmul(
            x_base, tf.matmul(diag, x_base, transpose_b=True)
        )
        nabla_11 = -0.5 * tf.matmul(
            y_base, tf.matmul(diag, y_base, transpose_b=True)
        )

        # Derivative for the output of both networks.
        x_der = 2 * tf.matmul(nabla_00, x) + tf.matmul(nabla_01, y)
        y_der = 2 * tf.matmul(nabla_11, y) + tf.matmul(
            nabla_01, x, transpose_a=True
        )

        x_der = 1 / (batch_size - 1) * x_der
        y_der = 1 / (batch_size - 1) * y_der

        # Transpose back as the input y_pred was
        x_1d = tf.transpose(x_der)
        y_1d = tf.transpose(y_der)

        # Concatenate it again
        concat_derivatives = tf.concat([x_1d, y_1d], axis=-1)

        # Stops the gradient calculations of Tensorflow
        concat_derivatives = tf.stop_gradient(concat_derivatives)

        # With a minus because Tensorflow minimizes the loss-function
        loss_score = -concat_derivatives * y_pred

        return loss_score

    def loss_VAMP2_autograd(self, y_true, y_pred):

        """
        Calculates VAMP-2 score with respect to the network lobes.
        Same function as loss_VAMP2, but gradient is computed
        automatically by tensorflow. Added when tensorflow >=1.5
        introduced gradients for eigenvalue decomposition and SVD.

        Parameters
        ----------
        y_true: tensorflow tensor
            parameter not needed for the calculation, added to
            comply with Keras rules for loss functions format

        y_pred: tensorflow tensor with shape
                [batch_size, 2 * output_size]
            output of the two lobes of the network

        Returns
        -------
        loss_score: tensorflow tensor with shape
                    [batch_size, 2 * output_size].
            gradient of the VAMP-2 score

        """

        # Remove the mean from the data
        x, y, batch_size, output_size = self._prep_data(y_pred)

        # Calculate the covariance matrices
        cov_01 = 1 / (batch_size - 1) * tf.matmul(x, y, transpose_b=True)
        cov_00 = 1 / (batch_size - 1) * tf.matmul(x, x, transpose_b=True)
        cov_11 = 1 / (batch_size - 1) * tf.matmul(y, y, transpose_b=True)

        # Calculate the inverse of the self-covariance matrices
        cov_00_inv = self._inv(cov_00, ret_sqrt=True)
        cov_11_inv = self._inv(cov_11, ret_sqrt=True)

        vamp_matrix = tf.matmul(tf.matmul(cov_00_inv, cov_01), cov_11_inv)

        vamp_score = tf.norm(vamp_matrix)

        return -tf.square(vamp_score)

    def loss_VAMP2(self, y_true, y_pred):

        """
        Calculates the gradient of the VAMP-2 score calculated
        with respect to the network lobes. Shrinkage algorithm
        gurantees that auto-covariance matrices are positive definite
        and their inverse square-root exists. Can be used as a
        loss function for a keras model.

        Parameters
        ----------
        y_true: tensorflow tensor
            parameter not needed for the calculation,
            added to comply with Keras rules for loss
            functions format

        y_pred: tensorflow tensor with shape
               [batch_size, 2 * output_size]
            output of the two lobes of the network

        Returns
        -------
        loss_score: tensorflow tensor with shape
                   [batch_size, 2 * output_size].
            gradient of the VAMP-2 score

        """

        # Remove the mean from the data
        x, y, batch_size, output_size = self._prep_data(y_pred)

        # Calculate the covariance matrices
        cov_01 = 1 / (batch_size - 1) * tf.matmul(x, y, transpose_b=True)
        cov_10 = 1 / (batch_size - 1) * tf.matmul(y, x, transpose_b=True)
        cov_00 = 1 / (batch_size - 1) * tf.matmul(x, x, transpose_b=True)
        cov_11 = 1 / (batch_size - 1) * tf.matmul(y, y, transpose_b=True)

        # Calculate the inverse of the self-covariance matrices
        cov_00_inv = self._inv(cov_00)
        cov_11_inv = self._inv(cov_11)

        # Split the gradient computation in 2 parts for readability
        # These are reported as Eq. 10, 11 in the VAMPnets paper
        left_part_x = tf.matmul(cov_00_inv, tf.matmul(cov_01, cov_11_inv))
        left_part_y = tf.matmul(cov_11_inv, tf.matmul(cov_10, cov_00_inv))

        right_part_x = y - tf.matmul(cov_10, tf.matmul(cov_00_inv, x))
        right_part_y = x - tf.matmul(cov_01, tf.matmul(cov_11_inv, y))

        # Calculate the dot product of the two matrices
        x_der = 2 / (batch_size - 1) * tf.matmul(left_part_x, right_part_x)
        y_der = 2 / (batch_size - 1) * tf.matmul(left_part_y, right_part_y)

        # Transpose back as the input y_pred was
        x_1d = tf.transpose(x_der)
        y_1d = tf.transpose(y_der)

        # Concatenate it again
        concat_derivatives = tf.concat([x_1d, y_1d], axis=-1)

        # Stop the gradient calculations of Tensorflow
        concat_derivatives = tf.stop_gradient(concat_derivatives)

        # With a minus because Tensorflow maximizes the loss-function
        loss_score = -concat_derivatives * y_pred

        return loss_score

    def metric_VAMP(self, y_true, y_pred):

        """
        Returns the sum of top k eigenvalues of the
        VAMP matrix, with k determined by the wrapper
        parameter k_eig, and the vamp matrix defined
        as:
            V = cov_00 ^ -1/2 * cov_01 * cov_11 ^ -1/2
        Can be used as a metric function in model.fit()

        Parameters
        ----------
        y_true: tensorflow tensor
            parameter not needed for the calculation,
            added to comply with Keras rules for loss functions
            format.

        y_pred: tensorflow tensor with shape
                [batch_size, 2 * output_size]
            output of the two lobes of the network

        Returns
        -------
        eig_sum: tensorflow float
            sum of the k highest eigenvalues in
            the vamp matrix

        """

        # Remove the mean from the data
        x, y, batch_size, output_size = self._prep_data(y_pred)

        # Calculate the inverse root of the auto-covariance matrices, and the
        # cross-covariance matrix
        matrices = self._build_vamp_matrices(x, y, batch_size)
        cov_00_ir, cov_11_ir, cov_01 = matrices

################ VAMPnet ################

        # Calculate the VAMP matrix
        vamp_matrix = tf.matmul(cov_00_ir, tf.matmul(cov_01, cov_11_ir))

        # Select the K highest singular values of the VAMP matrix
        diag = tf.convert_to_tensor(
            tf.linalg.svd(vamp_matrix, compute_uv=False)
        )
        cond = tf.greater(self.k_eig, 0)
        top_k_val = tf.nn.top_k(diag, k=self.k_eig)[0]

        # Sum the singular values
        eig_sum = tf.cond(
            cond, lambda: tf.reduce_sum(top_k_val), lambda: tf.reduce_sum(diag)
        )

        return eig_sum

    def metric_VAMP2(self, y_true, y_pred):

        """
        Returns the sum of squared top k eigenvalues of
        the VAMP matrix, with k determined by the wrapper
        parameter k_eig, and the vamp matrix
        defined as:
            V = cov_00 ^ -1/2 * cov_01 * cov_11 ^ -1/2
        Can be used as a metric function in model.fit()

        Parameters
        ----------
        y_true: tensorflow tensor
            parameter not needed for the calculation,
            added to comply with Keras rules for loss
            functions format

        y_pred: tensorflow tensor with shape
                [batch_size, 2 * output_size]
            output of the two lobes of the network

        Returns
        -------
        eig_sum_sq: tensorflow float
            sum of the squared k highest eigenvalues
            in the vamp matrix

        """

        # Remove the mean from the data
        x, y, batch_size, output_size = self._prep_data(y_pred)

        # Calculate the inverse root of the auto-covariance matrices, and the
        # cross-covariance matrix
        matrices = self._build_vamp_matrices(x, y, batch_size)
        cov_00_ir, cov_11_ir, cov_01 = matrices

        # Calculate the VAMP matrix
        vamp_matrix = tf.matmul(cov_00_ir, tf.matmul(cov_01, cov_11_ir))

        # Select the K highest singular values of the VAMP matrix
        diag = tf.convert_to_tensor(
            tf.linalg.svd(vamp_matrix, compute_uv=False)
        )
        cond = tf.greater(self.k_eig, 0)
        top_k_val = tf.nn.top_k(diag, k=self.k_eig)[0]

        # Square the singular values and sum them
        pow2_topk = tf.reduce_sum(tf.multiply(top_k_val, top_k_val))
        pow2_diag = tf.reduce_sum(tf.multiply(diag, diag))
        eig_sum_sq = tf.cond(cond, lambda: pow2_topk, lambda: pow2_diag)

        return eig_sum_sq

    def estimate_koopman_op(self, trajs, tau):

        """
        Estimates the koopman operator for a given
        trajectory at the lag time specified. Formula
        for the estimation is:
                K = C00 ^ -1 @ C01

        Parameters
        ----------
        traj: numpy array with size
              [traj_timesteps, traj_dimensions]
              or a list of trajectories
            Trajectory described by the returned
            koopman operator

        tau: int
            Time shift at which the koopman operator
            is estimated

        Returns
        -------
        koopman_op: numpy array with shape
                    [traj_dimensions, traj_dimensions]
            Koopman operator estimated at timeshift tau

        """

        if type(trajs) == list:
            traj = np.concatenate([t[:-tau] for t in trajs], axis=0)
            traj_lag = np.concatenate([t[tau:] for t in trajs], axis=0)
        else:
            traj = trajs[:-tau]
            traj_lag = trajs[tau:]

        c_0 = np.transpose(traj) @ traj
        c_tau = np.transpose(traj) @ traj_lag

        eigv, eigvec = np.linalg.eig(c_0)
        include = eigv > self._epsilon
        eigv = eigv[include]
        eigvec = eigvec[:, include]
        c0_inv = eigvec @ np.diag(1 / eigv) @ np.transpose(eigvec)

        koopman_op = c0_inv @ c_tau
        return koopman_op

    def get_its(self, traj, lags):

        """
        Implied timescales from a trajectory estimated at a
        series of lag times

        Parameters
        ----------
        traj: numpy array with size
              [traj_timesteps, traj_dimensions]
            trajectory data or a list of trajectories

        lags: numpy array with size [lag_times]
            series of lag times at which the implied
            timescales are estimated

        Returns
        -------
        its: numpy array with size
            [traj_dimensions - 1, lag_times]
            Implied timescales estimated for the trajectory.

        """

        if type(traj) == list:
            outputsize = traj[0].shape[1]
        else:
            outputsize = traj.shape[1]
        its = np.zeros((outputsize - 1, len(lags)))

        for t, tau_lag in enumerate(lags):
            koopman_op = self.estimate_koopman_op(traj, tau_lag)
            k_eigvals, k_eigvec = np.linalg.eig(np.real(koopman_op))
            k_eigvals = np.sort(np.absolute(k_eigvals))
            k_eigvals = k_eigvals[:-1]
            its[:, t] = -tau_lag / np.log(k_eigvals)

        return its

    def get_ck_test(self, traj, steps, tau):

        """
        Chapman-Kolmogorov test for koopman operator estimated
        for the given trajectory at the given lag times

        Parameters
        ----------
        traj: numpy array with size
              [traj_timesteps, traj_dimensions]
            trajectory data or a list of trajectories

        steps: int
            how many lag times the ck test will be evaluated at

        tau: int
            shift between consecutive lag times

        Returns
        -------
        predicted: numpy array with size
                  [traj_dimensions, traj_dimensions, steps]
        estimated: numpy array with size
                  [traj_dimensions, traj_dimensions, steps]
            Predicted and estimated transition probabilities at the
            indicated lag times

        """

        if type(traj) == list:
            n_states = traj[0].shape[1]
        else:
            n_states = traj.shape[1]

        predicted = np.zeros((n_states, n_states, steps))
        estimated = np.zeros((n_states, n_states, steps))

        predicted[:, :, 0] = np.identity(n_states)
        estimated[:, :, 0] = np.identity(n_states)

        for vector, i in zip(np.identity(n_states), range(n_states)):
            for n in range(1, steps):

                koop = self.estimate_koopman_op(traj, tau)

                koop_pred = np.linalg.matrix_power(koop, n)

                koop_est = self.estimate_koopman_op(traj, tau * n)

                predicted[i, :, n] = vector @ koop_pred
                estimated[i, :, n] = vector @ koop_est

        return [predicted, estimated]

    def estimate_koopman_constrained(self, traj, tau, th=0):

        """
        Calculate the transition matrix that minimizes the norm of
        prediction error between the trajectory and the tau-shifted
        trajectory, using estimate of the non-reversible koopman
        operator as a starting value. Constraints impose that all
        values in the matrix are positive, and that the row sum
        equals 1. It is achieved using a COBYLA scipy minimizer.

        Parameters
        ----------
        traj: numpy array with size
              [traj_timesteps, traj_dimensions]
            Trajectory described by the returned koopman operator

        tau: int
            Time shift at which koopman operator is estimated

        th: float, optional, default = 0
            Parameter used to force elements of the matrix
            to be higher than 0. Useful to prevent elements
            of the matrix to have small negative value
            due to numerical issues

        Returns
        -------
        koop_positive: numpy array with shape
                       [traj_dimensions, traj_dimensions]
            Koopman operator estimated at timeshift tau

        """

        if type(traj) == list:
            raise Error("Multiple trajectories not supported")

        koop_init = self.estimate_koopman_op(traj, tau)

        n_states = traj.shape[1]

        rs = lambda k: np.reshape(k, (n_states, n_states))

        def errfun(k):
            diff_matrix = traj[tau:].T - rs(k) @ traj[:-tau].T
            return np.linalg.norm(diff_matrix)

        constr = []

        for n in range(n_states ** 2):
            # elements > 0
            constr.append(
                {"type": "ineq", "fun": lambda x, n=n: x.flatten()[n] - th}
            )
            # elements < 1
            constr.append(
                {"type": "ineq", "fun": lambda x, n=n: 1 - x.flatten()[n] - th}
            )

        for n in range(n_states):
            # row sum < 1
            constr.append(
                {
                    "type": "ineq",
                    "fun": lambda x, n=n: 1
                    - np.sum(x.flatten()[n : n + n_states]),
                }
            )
            # row sum > 1
            constr.append(
                {
                    "type": "ineq",
                    "fun": lambda x, n=n: np.sum(x.flatten()[n : n + n_states])
                    - 1,
                }
            )

        koop_positive = scipy.optimize.minimize(
            errfun,
            koop_init,
            constraints=constr,
            method="COBYLA",
            tol=1e-10,
            options={"disp": False, "maxiter": 1e5},
        ).x

        return koop_positive

    def plot_its(self, its, lag, ylog=False):

        """
        Plots the implied timescales calculated by the function
        'get_its'

        Parameters
        ----------
        its: numpy array
            the its array returned by the function get_its

        lag: numpy array
            lag times array used to estimate the implied timescales

        ylog: Boolean, optional, default = False
            if true, the plot will be a logarithmic plot,
            otherwise it will be a semilogy plot

        """

        if ylog:
            plt.loglog(lag, its.T[:, ::-1])
            plt.loglog(lag, lag, "k")
            plt.fill_between(lag, lag, 0.99, alpha=0.2, color="k")
        else:
            plt.semilogy(lag, its.T[:, ::-1])
            plt.semilogy(lag, lag, "k")
            plt.fill_between(lag, lag, 0.99, alpha=0.2, color="k")
        plt.show()

    def plot_ck_test(self, pred, est, n_states, steps, tau):

        """
        Plots the result of the Chapman-Kolmogorov test calculated
        by the function 'get_ck_test'

        Parameters
        ----------
        pred: numpy array

        est: numpy array
            pred, est are the two arrays returned by the
            function get_ck_test

        n_states: int

        steps: int

        tau: int
            values used for the Chapman-Kolmogorov test as
            parameters in the function get_ck_test

        """

        fig, ax = plt.subplots(n_states, n_states, sharex=True, sharey=True)
        for index_i in range(n_states):
            for index_j in range(n_states):

                ax[index_i][index_j].plot(
                    range(0, steps * tau, tau),
                    pred[index_i, index_j],
                    color="b",
                )

                ax[index_i][index_j].plot(
                    range(0, steps * tau, tau),
                    est[index_i, index_j],
                    color="r",
                    linestyle="--",
                )

                ax[index_i][index_j].set_title(
                    str(index_i + 1) + "->" + str(index_j + 1),
                    fontsize="small",
                )

        ax[0][0].set_ylim((-0.1, 1.1))
        ax[0][0].set_xlim((0, steps * tau))
        ax[0][0].axes.get_xaxis().set_ticks(
            np.round(np.linspace(0, steps * tau, 3))
        )
        plt.show()

    def _inv(self, x, ret_sqrt=False):

        """
        Utility function that returns inverse of a matrix, with the
        option to return the square root of the inverse matrix.

        Parameters
        ----------
        x: numpy array with shape [m,m]
            matrix to be inverted

        ret_sqrt: bool, optional, default = False
            if True, the square root of the inverse matrix
            is returned instead

        Returns
        -------
        x_inv: numpy array with shape [m,m]
            inverse of the original matrix

        """

        # Calculate eigvalues and eigvectors
        eigval_all, eigvec_all = tf.linalg.eigh(x)

        # Filter out eigvalues below threshold and corresponding eigvectors
        eig_th = tf.constant(self.epsilon, dtype=tf.float32)
        index_eig = tf.cast(eigval_all > eig_th, dtype=tf.int32)
        _, eigval = tf.dynamic_partition(eigval_all, index_eig, 2)
        _, eigvec = tf.dynamic_partition(
            tf.transpose(eigvec_all), index_eig, 2
        )

        # Build the diagonal matrix with the filtered eigenvalues or square
        # root of the filtered eigenvalues according to the parameter
        eigval_inv = tf.linalg.diag(1 / eigval)
        eigval_inv_sqrt = tf.linalg.diag(tf.sqrt(1 / eigval))

        cond_sqrt = tf.convert_to_tensor(ret_sqrt)

        diag = tf.cond(cond_sqrt, lambda: eigval_inv_sqrt, lambda: eigval_inv)

        # Rebuild the square root of the inverse matrix
        x_inv = tf.matmul(tf.transpose(eigvec), tf.matmul(diag, eigvec))

        return x_inv

    def _prep_data(self, data):

        """
        Utility function that transforms the input data from a tensorflow -
        viable format to a structure used by the following functions in
        the pipeline

        Parameters
        ----------
        data: tensorflow tensor with shape [b, 2*o]
            original format of the data

        Returns
        -------
        x: tensorflow tensor with shape [o, b]
            transposed, mean-free data corresponding
            to the left, lag-free lobe of the network

        y: tensorflow tensor with shape [o, b]
            transposed, mean-free data corresponding to
            the right, lagged lobe of the network

        b: tensorflow float32
            batch size of the data

        o: int
            output size of each lobe of the network

        """

        shape = tf.shape(data)
        b = tf.cast(shape[0], dtype=tf.float32)
        o = shape[1] // 2

        # Split the data of the two networks and transpose it
        x_biased = tf.transpose(data[:, :o])
        y_biased = tf.transpose(data[:, o:])

        # Subtract the mean
        x = x_biased - tf.reduce_mean(x_biased, axis=1, keepdims=True)
        y = y_biased - tf.reduce_mean(y_biased, axis=1, keepdims=True)

        return x, y, b, o

    def _build_vamp_matrices(self, x, y, b):

        """
        Utility function that returns the matrices used to compute
        VAMP scores and their gradients for non-reversible problems

        Parameters
        ----------
        x: tensorflow tensor with shape [output_size, b]
            output of the left lobe of the network

        y: tensorflow tensor with shape [output_size, b]
            output of the right lobe of the network

        b: tensorflow float32
            batch size of the data

        Returns
        -------
        cov_00_inv_root: numpy array with shape
                         [output_size, output_size]
            square root of the inverse of the auto-covariance
            matrix of x

        cov_11_inv_root: numpy array with shape
                         [output_size, output_size]
            square root of the inverse of the auto-covariance
            matrix of y

        cov_01: numpy array with shape
                [output_size, output_size]
            cross-covariance matrix of x and y

        """

        # Calculate the cross-covariance
        cov_01 = 1 / (b - 1) * tf.matmul(x, y, transpose_b=True)
        # Calculate the auto-correations
        cov_00 = 1 / (b - 1) * tf.matmul(x, x, transpose_b=True)
        cov_11 = 1 / (b - 1) * tf.matmul(y, y, transpose_b=True)

        # Calculate the inverse root of the auto-covariance
        cov_00_inv_root = self._inv(cov_00, ret_sqrt=True)
        cov_11_inv_root = self._inv(cov_11, ret_sqrt=True)

        return cov_00_inv_root, cov_11_inv_root, cov_01

    def _build_vamp_matrices_rev(self, x, y, b):

        """
        Utility function that returns the matrices used to compute VAMP
        scores and their gradients for reversible problems. Matrices are
        transformed into symmetrical matrices by calculating covariances
        using the mean of the auto- and cross-covariances, so that:
            cross_cov = 1/2*(cov_01 + cov_10)
        and:
            auto_cov = 1/2*(cov_00 + cov_11)


        Parameters
        ----------
        x: tensorflow tensor with shape [output_size, b]
            output of the left lobe of the network

        y: tensorflow tensor with shape [output_size, b]
            output of the right lobe of the network

        b: tensorflow float32
            batch size of the data

        Returns
        -------
        auto_cov_inv_root: numpy array with shape
                           [output_size, output_size]
            square root of the inverse of mean over the
            auto-covariance matrices of x and y

        cross_cov: numpy array with shape
                   [output_size, output_size]
            mean of the cross-covariance matrices of x and y

        """

        # Calculate the cross-covariances
        cov_01 = 1 / (b - 1) * tf.matmul(x, y, transpose_b=True)
        cov_10 = 1 / (b - 1) * tf.matmul(y, x, transpose_b=True)
        cross_cov = 1 / 2 * (cov_01 + cov_10)
        # Calculate the auto-covariances
        cov_00 = 1 / (b - 1) * tf.matmul(x, x, transpose_b=True)
        cov_11 = 1 / (b - 1) * tf.matmul(y, y, transpose_b=True)
        auto_cov = 1 / 2 * (cov_00 + cov_11)

        # Calculate the inverse root of the auto-covariance
        auto_cov_inv_root = self._inv(auto_cov, ret_sqrt=True)

        return auto_cov_inv_root, cross_cov

################ Experimental Functions on VAMPnet ################

    def _loss_VAMP_sym(self, y_true, y_pred):

        """
        WORK IN PROGRESS

        Calculates gradient of the VAMP-1 score calculated with respect
        to the network lobes. Shrinkage algorithm guarantees that
        auto-covariance matrices are positive definite and their
        inverse square-root exists. Can be used as a loss function for
        a keras model. The difference with the main loss_VAMP function
        is that here the matrices C00, C01, C11 are 'mixed' together:

        C00' = C11' = (C00+C11)/2
        C01 = C10 = (C01 + C10)/2

        There is no mathematical reasoning behind this experimental
        loss function. It performs worse than VAMP-2 with regard to
        the identification of processes, but it also helps the network
        to converge to a transformation that separates more neatly the
        different states

        Parameters
        ----------
        y_true: tensorflow tensor
            parameter not needed for the calculation,
            added to comply with Keras rules for loss
            functions format.

        y_pred: tensorflow tensor with shape
                [batch_size, 2 * output_size]
            output of the two lobes of the network

        Returns
        -------
        loss_score: tensorflow tensor with shape
                    [batch_size, 2 * output_size].
            gradient of the VAMP-1 score

        """

        # Remove the mean from the data
        x, y, batch_size, output_size = self._prep_data(y_pred)

        # Calculate the inverse root of the auto-covariance matrix, and the
        # cross-covariance matrix
        cov_00_ir, cov_01 = self._build_vamp_matrices_rev(x, y, batch_size)

        vamp_matrix = tf.matmul(cov_00_ir, tf.matmul(cov_01, cov_00_ir))

        D, U, V = tf.linalg.svd(vamp_matrix, full_matrices=True)
        diag = tf.linalg.diag(D)

        # Base-changed covariance matrices
        x_base = tf.matmul(cov_00_ir, U)
        y_base = tf.matmul(V, cov_00_ir, transpose_a=True)

        # Derivative for the output of both networks.
        nabla_01 = tf.matmul(x_base, y_base)
        nabla_00 = -0.5 * tf.matmul(
            x_base, tf.matmul(diag, x_base, transpose_b=True)
        )

        # Derivative for the output of both networks.
        x_der = (
            2
            / (batch_size - 1)
            * (tf.matmul(nabla_00, x) + tf.matmul(nabla_01, y))
        )
        y_der = (
            2
            / (batch_size - 1)
            * (tf.matmul(nabla_00, y) + tf.matmul(nabla_01, x))
        )

        # Transpose back as the input y_pred was
        x_1d = tf.transpose(x_der)
        y_1d = tf.transpose(y_der)

        # Concatenate it again
        concat_derivatives = tf.concat([x_1d, y_1d], axis=-1)

        # Stop the gradient calculations of Tensorflow
        concat_derivatives = tf.stop_gradient(concat_derivatives)

        # With a minus because Tensorflow maximizes the loss-function
        loss_score = -concat_derivatives * y_pred

        return loss_score

    def _metric_VAMP_sym(self, y_true, y_pred):

        """
        Metric function relative to the _loss_VAMP_sym function.

        Parameters
        ----------
        y_true: tensorflow tensor.
            parameter not needed for the calculation, added to
            comply with Keras rules for loss functions format.

        y_pred: tensorflow tensor with shape
                [batch_size, 2 * output_size]
            output of the two lobes of the network

        Returns
        -------
        eig_sum: tensorflow float
            sum of the k highest eigenvalues in the vamp matrix

        """

        # Remove the mean from the data
        x, y, batch_size, output_size = self._prep_data(y_pred)

        # Calculate the inverse root of the auto-covariance matrices, and the
        # cross-covariance matrix
        cov_00_ir, cov_01 = self._build_vamp_matrices_rev(x, y, batch_size)

        # Calculate the VAMP matrix
        vamp_matrix = tf.matmul(cov_00_ir, tf.matmul(cov_01, cov_00_ir))

        # Select the K highest singular values of the VAMP matrix
        diag = tf.convert_to_tensor(
            tf.linalg.svd(vamp_matrix, compute_uv=False)
        )
        cond = tf.greater(self.k_eig, 0)
        top_k_val = tf.nn.top_k(diag, k=self.k_eig)[0]

        # Sum the singular values
        eig_sum = tf.cond(
            cond, lambda: tf.reduce_sum(top_k_val), lambda: tf.reduce_sum(diag)
        )

        return eig_sum

    def _estimate_koopman_op(self, traj, tau):

        """
        Estimates the koopman operator for a given trajectory
        at the lag time specified. The formula for the
        estimation is:
        K = C00 ^ -1/2 @ C01 @ C11 ^ -1/2

        Parameters
        ----------
        traj: numpy array with size [traj_timesteps, traj_dimensions]
            Trajectory described by the returned koopman operator

        tau: int
            Time shift at which the koopman operator is estimated

        Returns
        -------
        koopman_op: numpy array with shape
            [traj_dimensions, traj_dimensions]
            Koopman operator estimated at timeshift tau

        """

        c_0 = traj[:-tau].T @ traj[:-tau]
        c_1 = traj[tau:].T @ traj[tau:]
        c_tau = traj[:-tau].T @ traj[tau:]

        eigv0, eigvec0 = np.linalg.eig(c_0)
        include0 = eigv0 > self._epsilon
        eigv0_root = np.sqrt(eigv0[include0])
        eigvec0 = eigvec0[:, include0]
        c0_inv_root = eigvec0 @ np.diag(1 / eigv0_root) @ eigvec0.T

        eigv1, eigvec1 = np.linalg.eig(c_1)
        include1 = eigv1 > self._epsilon
        eigv1_root = np.sqrt(eigv1[include1])
        eigvec1 = eigvec1[:, include1]
        c1_inv_root = eigvec1 @ np.diag(1 / eigv1_root) @ eigvec1.T

        koopman_op = c0_inv_root @ c_tau @ c1_inv_root
        return koopman_op

################ Experimental Functions on VAMPnet ################

################ VAMPnet ################
