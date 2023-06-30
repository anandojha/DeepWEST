from scipy.ndimage.filters import gaussian_filter
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from math import sin 
from math import cos 
import mdtraj as md
import pandas as pd
import numpy as np
import itertools
import sys
import os
pwd_deepwest = "/home/aaojha/DeepWEST/"  # PWD of the directory where DeepWEST is installed
path_join = pwd_deepwest + "DeepWEST/"
module_path = os.path.abspath(os.path.join(path_join))
if module_path not in sys.path:
    sys.path.append(module_path)
import DeepWEST 
DeepWEST.get_chignolin_ref_pdb()
# Load Data ( .prmtop and .nc should be present)
data_dir = os.getcwd()
traj_file = os.path.join(data_dir, "system_final_100ns.nc")
top = os.path.join(data_dir, "system_final.prmtop")
heavy_atoms_file = os.path.join("heavy_atoms_md_chignolin.txt")
rmsd_rg_file = os.path.join("rmsd_rg_md_chignolin.txt")
dihed1_dihed2_file = os.path.join("dihed1_dihed2_md_chignolin.txt")
distance_rg_file = os.path.join("distance_rg_md_chignolin.txt")
# Define Parameters and Hyperparameters
attempts = 1 #10
start = 0 #0
stop = 500000
stride = 1 #1
no_frames = 50 # Number of frames to be selected from each bin in the output state
output_size =  3 # How many output states the network has (max = 6)
tau = 200 #Tau, how much is the timeshift of the two datase
batch_size = 1000 # Batch size for Stochastic Gradient descent
train_ratio = 0.9 # Which trajectory points percentage is used as training
network_depth = 6 # How many hidden layers the network has
layer_width = 100 # Width of every layer
learning_rate = 1e-4 # Learning rate used for the ADAM optimizer
nb_epoch = 100 # Iteration over the training set in the fitting process
epsilon = 1e-5 # epsilon 
# Define data points
DeepWEST.create_heavy_atom_xyz_no_solvent(traj = traj_file, top = top, heavy_atoms_array = heavy_atoms_file, start = start, stop = stop, stride = stride)
DeepWEST.create_rmsd_rg_chignolin_top(traj = traj_file, top = top, rmsd_rg_txt = rmsd_rg_file, start = start, stop = stop, stride = stride)
#scheme = 'ca', 'closest', 'closest-heavy'
DeepWEST.create_input_features_chignolin(traj=traj_file, top=top, scheme = 'closest-heavy', feat_txt=dihed1_dihed2_file, start = start, stop = stop, stride = stride)

traj_whole = np.loadtxt(heavy_atoms_file)
print(traj_whole.shape)
dihed1_dihed2 = np.loadtxt(dihed1_dihed2_file)
print(dihed1_dihed2.shape)
rmsd_rg = np.loadtxt(rmsd_rg_file)
print(rmsd_rg.shape)
traj_data_points, input_size = traj_whole.shape
# Initialized the VAMPnets wrapper class
vamp = DeepWEST.VampnetTools(epsilon = epsilon)
# Shuffle trajectory and lagged trajectory together
length_data = traj_data_points - tau
traj_ord = traj_whole[:length_data]
traj_ord_lag = traj_whole[tau : length_data + tau]
dihed1_dihed2_init = dihed1_dihed2[:length_data]
rmsd_rg_init = rmsd_rg[:length_data]
indexes = np.arange(length_data)
np.random.shuffle(indexes)
shuff_indexes = indexes.copy()
traj = traj_ord[indexes]
traj_lag = traj_ord_lag[indexes]
dihed1_dihed2_shuffle = dihed1_dihed2_init[indexes]
# Prepare data for tensorflow usage
length_train = int(np.floor(length_data * train_ratio))
length_vali = length_data - length_train
traj_data_train = traj[:length_train]
traj_data_train_lag = traj_lag[:length_train]
traj_data_valid = traj[length_train:]
traj_data_valid_lag = traj_lag[length_train:]
# Input of the first network
X1_train = traj_data_train.astype('float32')
X2_train  = traj_data_train_lag.astype('float32')
# Input for validation
X1_vali = traj_data_valid.astype('float32')
X2_vali = traj_data_valid_lag.astype('float32')
# Needs a Y-train set which we dont have.
Y_train = np.zeros((length_train,2*output_size)).astype('float32')
Y_vali = np.zeros((length_vali,2*output_size)).astype('float32')
# Run several model iterations saving the best one, to help finding sparcely populated states
max_vm = 0
attempts = attempts
losses = [vamp.loss_VAMP2_autograd]
for i in range(attempts):    
    # Clear the previous tensorflow session to prevent memory leaks
    #clear_session()
    tf.keras.backend.clear_session()
    # Build the model
    nodes = [layer_width]*network_depth
    Data_X = tf.keras.layers.Input(shape = (input_size,))
    Data_Y = tf.keras.layers.Input(shape = (input_size,))
    # Batch normalization layer improves convergence speed
    bn_layer = tf.keras.layers.BatchNormalization()
    # Instance layers and assign them to the two lobes of the network
    dense_layers = [tf.keras.layers.Dense(node, activation = 'elu') 
                    # if index_layer < 3 else 'linear nodes')
                    for index_layer,node in enumerate(nodes)]
    lx_branch = bn_layer(Data_X)
    rx_branch = bn_layer(Data_Y)
    for i, layer in enumerate(dense_layers):
        lx_branch = dense_layers[i](lx_branch)
        rx_branch = dense_layers[i](rx_branch)
    # Add a softmax output layer
    # Should be replaced with a linear activation layer if
    # the outputs of the network cannot be interpreted as states
    softmax = tf.keras.layers.Dense(output_size, activation='softmax')
    lx_branch = softmax(lx_branch)
    rx_branch = softmax(rx_branch)
    # Merge both networks to train both at the same time
    merged = tf.keras.layers.concatenate([lx_branch, rx_branch])
    # Initialize the model and the optimizer, and compile it with
    # the loss and metric functions from the VAMPnets package
    model = tf.keras.models.Model(inputs = [Data_X, Data_Y], outputs = merged)
    adam = tf.keras.optimizers.Adam(learning_rate = learning_rate/10)
    vm1 = np.zeros((len(losses), nb_epoch))
    tm1 = np.zeros_like(vm1)
    vm2 = np.zeros_like(vm1)
    tm2 = np.zeros_like(vm1)
    vm3 = np.zeros_like(vm1)
    tm3 = np.zeros_like(vm1)
    for l_index, loss_function in enumerate(losses):
        model.compile(optimizer = adam,
                      loss = loss_function,
                      metrics = [vamp.metric_VAMP, vamp.metric_VAMP2])
        # Train the model  
        hist = model.fit([X1_train, X2_train], Y_train ,
                         batch_size=batch_size,
                         epochs=nb_epoch,
                         validation_data=([X1_vali, X2_vali], Y_vali ),
                         verbose=0)
        vm1[l_index] = np.array(hist.history['val_metric_VAMP'])
        tm1[l_index] = np.array(hist.history['metric_VAMP'])
        vm2[l_index] = np.array(hist.history['val_metric_VAMP2'])
        tm2[l_index] = np.array(hist.history['metric_VAMP2'])
        vm3[l_index] = np.array(hist.history['val_loss'])
        tm3[l_index] = np.array(hist.history['loss'])
    vm1 = np.reshape(vm1, (-1))
    tm1 = np.reshape(tm1, (-1))
    vm2 = np.reshape(vm2, (-1))
    tm2 = np.reshape(tm2, (-1))
    vm3 = np.reshape(vm3, (-1))
    tm3 = np.reshape(tm3, (-1))
    # Average the score obtained in the last part of the training process
    # in order to establish which model is better and thus worth saving
    score = vm1[-5:].mean()
    extra_msg = ''
    if score > max_vm:
        extra_msg = ' - Highest'
        best_weights = model.get_weights()
        max_vm = score
        vm1_max = vm1
        tm1_max = tm1
        vm2_max = vm2
        tm2_max = tm2
        vm3_max = vm3
        tm3_max = tm3  
    print('Score: {0:.2f}'.format(score) + extra_msg)
# Recover the saved model and its training history
model.set_weights(best_weights)
tm1 = np.array(tm1_max)
tm2 = np.array(tm2_max)
tm3 = np.array(tm3_max)
vm1 = np.array(vm1_max)
vm2 = np.array(vm2_max)
vm3 = np.array(vm3_max)
# Training result visualization
plt.figure(figsize=(12,8))
plt.plot(vm1, label = 'VAMP')
plt.plot(vm2, label = 'VAMP2')
plt.plot(-vm3, label = 'loss')
plt.plot(tm1, label = 'training VAMP')
plt.plot(tm2, label = 'training VAMP2')
plt.plot(-tm3, label = 'training loss')
plt.legend()
plt.savefig("rates_chignolin.jpg", bbox_inches="tight", dpi = 500)
plt.show(block=False)
plt.pause(1)
plt.close()
# Transform the input trajectory using the network
states_prob = model.predict([traj_ord, traj_ord_lag])[:, :output_size]
# Order the output states based on their population
coor_pred = np.argmax(states_prob, axis = 1)
indexes = [np.where(coor_pred == np.multiply(np.ones_like(coor_pred), n)) 
           for n in range(output_size)]
states_num = [len(i[0]) for i in indexes]
states_order = np.argsort(states_num).astype('int')[::-1]
pred_ord = states_prob[:,states_order]
# Visualize the population of the states
def print_states_pie_chart():
    coors = []
    maxi = np.max(pred_ord, axis= 1)
    for i in range(output_size):
        coors.append(len(np.where(pred_ord[:,i] == maxi)[0]))
    fig1, ax1 = plt.subplots()
    ax1.pie(np.array(coors), autopct='%1.2f%%', startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    print('States population: '+ str(np.array(coors)/len(maxi)*100)+'%')
    np.savetxt('population.txt', coors, delimiter=',')
    plt.savefig("population_chignolin.jpg", bbox_inches="tight", dpi = 500)
    plt.show(block=False)
    plt.pause(1)
    plt.close()
print_states_pie_chart()
# Visualize how the states are placed on the Ramachandran plot
maxi_train = np.max(pred_ord, axis= 1)
coor_train = np.zeros_like(pred_ord)
for i in range(output_size):
    coor_train = np.where(pred_ord[:,i]== maxi_train)[0]
    plt.scatter(dihed1_dihed2_init[coor_train,0], dihed1_dihed2_init[coor_train,1], s=5)
    plt.savefig("dist_chignolin.jpg", dpi = 500)
    plt.axes = [[-np.pi, np.pi],[-np.pi, np.pi]]
# For each state, visualize the probabilities the different trajectory points have to belong to it
fig = plt.figure(figsize=(25,25))
gs1 = gridspec.GridSpec(2, int(np.ceil(output_size/2)))
gs1.update(wspace=0.05, hspace = 0.05)
for n in range(output_size):
    ax = plt.subplot(gs1[n])
    im = ax.scatter(dihed1_dihed2_init[:,0], dihed1_dihed2_init[:,1], s=30, c = pred_ord[:,n], alpha=0.5, vmin = 0, vmax = 1)
    plt.axis('on')
    title = 'State '+str(n + 1)
    ax.text(.85, .15, title,
        horizontalalignment='center',
        transform=ax.transAxes,  fontdict = {'size':36})
    if (n < 3):
        ax.set_xticks([-3, 0, 3])
        ax.set_xticklabels([r'-$\pi$', r'$0$', r'$\pi$'])
        ax.xaxis.set_tick_params(top='on', bottom='off', labeltop='on', labelbottom='off')
        ax.xaxis.set_tick_params(labelsize=40)
    else:
        ax.set_xticks([])
    if (n%3==0):
        ax.set_yticks([-3, 0, 3])
        ax.set_yticklabels([r'-$\pi$', r'$0$', r'$\pi$'])
        ax.yaxis.set_tick_params(labelsize=40)
    else:
        ax.set_yticks([])
#    ax.set_aspect('equal')
    ax.set_xlim([-np.pi, np.pi]);
    ax.set_ylim([-np.pi, np.pi]);
    if (n%3 == 0):
        ax.set_ylabel(r'$\Psi$ [rad]', fontdict = {'size':40})
    if (n < 3):
        ax.set_xlabel(r'$\Phi$ [rad]', fontdict = {'size':40}, position = 'top')
        ax.xaxis.set_label_coords(0.5,1.2)
gs1.tight_layout(fig, rect=[0, 0.03, 0.95, 0.94])
cax = fig.add_axes([0.95, 0.05, 0.02, 0.8])
cbar = fig.colorbar(im, cax=cax, ticks=[0, 1])
cbar.ax.yaxis.set_tick_params(labelsize=40)
fig.savefig("state_viz_chignolin.jpg", bbox_inches="tight", dpi = 250)
plt.close()
# Markov Model Estimation
"""
# Estimate the implied timescales
max_tau = 200
lag = np.arange(1, max_tau, 1)
its = vamp.get_its(pred_ord, lag)
vamp.plot_its(its, lag, fig = "its_chignolin.jpg")
"""
# Chapman-Kolmogorov test for the estimated koopman operator
steps = 8
tau_msm = 35
predicted, estimated = vamp.get_ck_test(pred_ord, steps, tau_msm)
# vamp.plot_ck_test(predicted, estimated, output_size, steps, tau_msm)
# Saving the frame indices to a txt file
indices_list = [idxs[0].tolist() for idxs in indexes]
print("Saving indices")
sorted_indices = DeepWEST.get_pdbs_from_clusters(indices = indices_list, rmsd_rg=dihed1_dihed2_shuffle, num_pdbs = no_frames)
print("Saved indices")
index_for_we = list(itertools.chain.from_iterable(sorted_indices))
print(len(index_for_we))
np.savetxt("indices_vamp_chignolin.txt", index_for_we)
current_cwd = os.getcwd()
westpa_cwd = current_cwd + "/" + "westpa_dir"  # westpa directory pwd
indices_vamp = np.loadtxt("indices_vamp_chignolin.txt")
indices_vamp = [int(i) for i in indices_vamp]

# Visualize how the states are placed on the Rg-RMSD plot  
maxi_train = np.max(pred_ord, axis= 1)
coor_train = np.zeros_like(pred_ord)
for i in range(0, output_size-2):
    coor_train = np.where(pred_ord[:,i]== maxi_train)[0]
    plt.scatter(rmsd_rg_init[coor_train,0], rmsd_rg_init[coor_train,1], s=8, color="green")
    plt.xlabel(r'RMSD (nm)')
    plt.ylabel(r'$R_g$ (nm)')
    plt.xlim([-0.05,1.05])
    plt.ylim([0.395,1.05])
    plt.savefig("dist_chignolin_rmsd_rg_0.jpg", dpi = 500)
    plt.show(block=False)
    plt.pause(3)
    plt.close()
for i in range(1, output_size-1):
    coor_train = np.where(pred_ord[:,i]== maxi_train)[0]
    plt.scatter(rmsd_rg_init[coor_train,0], rmsd_rg_init[coor_train,1], s=8, color="red")
    plt.xlabel(r'RMSD (nm)')
    plt.ylabel(r'$R_g$ (nm)')
    plt.xlim([-0.05,1.05])
    plt.ylim([0.395,1.05])
    plt.savefig("dist_chignolin_rmsd_rg_1.jpg", dpi = 500)
    plt.show(block=False)
    plt.pause(3)
    plt.close()
for i in range(2, output_size):
    coor_train = np.where(pred_ord[:,i]== maxi_train)[0]
    plt.scatter(rmsd_rg_init[coor_train,0], rmsd_rg_init[coor_train,1], s=8, color="blue")
    plt.xlabel(r'RMSD (nm)')
    plt.ylabel(r'$R_g$ (nm)')
    plt.xlim([-0.05,1.05])
    plt.ylim([0.395,1.05])
    plt.savefig("dist_chignolin_rmsd_rg_2.jpg", dpi = 500)
    plt.show(block=False)
    plt.pause(3)
    plt.close()    
for i in range(2, output_size):
    coor_train = np.where(pred_ord[:,i]== maxi_train)[0]
    plt.scatter(rmsd_rg_init[coor_train,0], rmsd_rg_init[coor_train,1], s=10, color="blue", marker="^", alpha=1.00)
    plt.xlabel(r'RMSD (nm)')
    plt.ylabel(r'$R_g$ (nm)')
    plt.xlim([-0.05,1.05])
    plt.ylim([0.395,1.05]) 
for i in range(0, output_size-2):
    coor_train = np.where(pred_ord[:,i]== maxi_train)[0]
    plt.scatter(rmsd_rg_init[coor_train,0], rmsd_rg_init[coor_train,1], s=10, color="green", marker='v', alpha = 1.00)
    plt.xlabel(r'RMSD (nm)')
    plt.ylabel(r'$R_g$ (nm)')
    plt.xlim([-0.05,1.05])
    plt.ylim([0.395,1.05])  
for i in range(1, output_size-1):
    coor_train = np.where(pred_ord[:,i]== maxi_train)[0]
    plt.scatter(rmsd_rg_init[coor_train,0], rmsd_rg_init[coor_train,1], s=10, color="red", marker='v', alpha = 1.00)
    plt.xlabel(r'RMSD (nm)')
    plt.ylabel(r'$R_g$ (nm)')
    plt.xlim([-0.05,1.05])
    plt.ylim([0.395,1.05])  
    plt.savefig("dist_chignolin_rmsd_rg_all.jpg", dpi = 500)
    plt.show(block=False)
    plt.pause(3)
    plt.close()

maxi_train = np.max(pred_ord, axis= 1)
coor_train = np.zeros_like(pred_ord)
for i in range(0, output_size-2):
    coor_train = np.where(pred_ord[:,i]== maxi_train)[0]
    x = rmsd_rg_init[coor_train,0]
    y = rmsd_rg_init[coor_train,1]
    x1_data=[]
    for i in x: 
        x1_data.append(i)
    y1_data=[]
    for j in y: 
        y1_data.append(j)
for i in range(1, output_size-1):
    coor_train = np.where(pred_ord[:,i]== maxi_train)[0]
    x = rmsd_rg_init[coor_train,0]
    y = rmsd_rg_init[coor_train,1]
    x2_data=[]
    for i in x: 
        x2_data.append(i)
    y2_data=[]
    for j in y: 
        y2_data.append(j) 
for i in range(2, output_size):
    coor_train = np.where(pred_ord[:,i]== maxi_train)[0]
    x = rmsd_rg_init[coor_train,0]
    y = rmsd_rg_init[coor_train,1]
    x3_data=[]
    for i in x: 
        x3_data.append(i)
    y3_data=[]
    for j in y: 
        y3_data.append(j) 
df_1 = pd.DataFrame(list(zip(x1_data, y1_data)))
df_1.columns=["x1","y1"]
df_2 = pd.DataFrame(list(zip(x2_data, y2_data)))
df_2.columns=["x2","y2"]
df_3 = pd.DataFrame(list(zip(x3_data, y3_data)))
df_3.columns=["x3","y3"]
df_1.to_csv("data1.csv", sep='\t',index=False)
df_2.to_csv("data2.csv", sep='\t',index=False)
df_3.to_csv("data3.csv", sep='\t',index=False)
x1_data = df_1['x1'].tolist()
y1_data = df_1['y1'].tolist()
x2_data = df_2['x2'].tolist()
y2_data = df_2['y2'].tolist()
x3_data = df_3['x3'].tolist()
y3_data = df_3['y3'].tolist()
x1_data.append(1.00)
y1_data.append(1.00)
x1_data.append(0.00)
y1_data.append(0.45)
x2_data.append(1.00)
y2_data.append(1.00)
x2_data.append(0.00)
y2_data.append(0.45)
x3_data.append(1.00)
y3_data.append(1.00)
x3_data.append(0.00)
y3_data.append(0.45)
n = 4
ticks = range(n)
colors = plt.cm.get_cmap('gist_heat_r',n)(ticks)
lcmap = plt.matplotlib.colors.ListedColormap(colors)
fig, (ax3, ax1, ax2) = plt.subplots(1, 3, figsize=(14, 6), dpi=2000, gridspec_kw={'width_ratios': [1.1, 1.1, 1]})
plt.subplots_adjust(wspace=0.4, hspace=0.4)
heatmap, xedges, yedges = np.histogram2d(x1_data, y1_data, bins=1000)
heatmap = gaussian_filter(heatmap, sigma=64)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
im1=ax1.imshow(heatmap.T, extent=extent, origin='lower', cmap=lcmap)
ax1.set_xlabel(r'RMSD (nm)', fontsize=12)
ax1.set_ylabel(r'$R_g$ (nm)', fontsize=12)
plt.setp(ax1.get_xticklabels(), fontsize=12)
plt.setp(ax1.get_yticklabels(), fontsize=12)
heatmap, xedges, yedges = np.histogram2d(x3_data, y3_data, bins=1000)
heatmap = gaussian_filter(heatmap, sigma=64)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
im3=ax3.imshow(heatmap.T, extent=extent, origin='lower', cmap=lcmap)
ax3.set_xlabel(r'RMSD (nm)', fontsize=12)
ax3.set_ylabel(r'$R_g$ (nm)', fontsize=12)
plt.setp(ax3.get_xticklabels(), fontsize=12)
plt.setp(ax3.get_yticklabels(), fontsize=12)
heatmap, xedges, yedges = np.histogram2d(x2_data, y2_data, bins=1000)
heatmap = gaussian_filter(heatmap, sigma=64)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
im2=ax2.imshow(heatmap.T, extent=extent, origin='lower', cmap=lcmap)
ax2.set_xlabel(r'RMSD (nm)', fontsize=12)
ax2.set_ylabel(r'$R_g$ (nm)', fontsize=12)
plt.setp(ax2.get_xticklabels(), fontsize=12)
plt.setp(ax2.get_yticklabels(), fontsize=12)
fig.colorbar(im1, ax=ax1,fraction=0.022, pad=0.1)
fig.colorbar(im2, ax=ax2,fraction=0.022, pad=0.1)
fig.colorbar(im3, ax=ax3,fraction=0.022, pad=0.1)
plt.savefig('chignolin_distributions.jpeg', bbox_inches='tight', pad_inches=0.2, dpi=500)

n = 1000
ticks = range(n)
colors = plt.cm.get_cmap('jet',n)(ticks)
lcmap = plt.matplotlib.colors.ListedColormap(colors)
#'jet', Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg','brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma','magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r','prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'

plt.figure(figsize=(8, 8), dpi=2000)
heatmap, xedges, yedges = np.histogram2d(x1_data, y1_data, bins=1000)
heatmap = gaussian_filter(heatmap, sigma=64)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.imshow(heatmap.T, extent=extent, origin='lower', cmap=lcmap)
plt.xlabel(r'RMSD (nm)', fontsize=12)
plt.ylabel(r'$R_g$ (nm)', fontsize=12)
cbar = plt.colorbar(fraction=0.0265)    
ticklabels = ['low', 'medium', 'high']
cbar.set_ticks(np.linspace(0.5, 3.50, len(ticklabels)))
cbar.set_ticklabels(ticklabels)
cbar.set_label('Frequency', rotation=270,loc='center', labelpad=15)
plt.savefig('chignolin_dist_1.jpeg', dpi=1000, bbox_inches = "tight")
plt.show()


plt.figure(figsize=(8, 8), dpi=2000)
heatmap, xedges, yedges = np.histogram2d(x2_data, y2_data, bins=1000)
heatmap = gaussian_filter(heatmap, sigma=64)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.imshow(heatmap.T, extent=extent, origin='lower', cmap=lcmap)
plt.xlabel(r'RMSD (nm)', fontsize=12)
plt.ylabel(r'$R_g$ (nm)', fontsize=12)
cbar = plt.colorbar(fraction=0.0265)    
ticklabels = ['low', 'medium', 'high']
cbar.set_ticks(np.linspace(0.35, 2.40, len(ticklabels)))
cbar.set_ticklabels(ticklabels)
cbar.set_label('Frequency', rotation=270,loc='center', labelpad=15)
plt.savefig('chignolin_dist_2.jpeg', dpi=1000, bbox_inches = "tight")
plt.show()

plt.figure(figsize=(8, 8), dpi=2000)
heatmap, xedges, yedges = np.histogram2d(x3_data, y3_data, bins=1000)
heatmap = gaussian_filter(heatmap, sigma=64)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.imshow(heatmap.T, extent=extent, origin='lower', cmap=lcmap)
plt.xlabel(r'RMSD (nm)', fontsize=12)
plt.ylabel(r'$R_g$ (nm)', fontsize=12)
cbar = plt.colorbar(fraction=0.0265)    
ticklabels = ['low', 'medium', 'high']
cbar.set_ticks(np.linspace(0.02, 0.20, len(ticklabels)))
cbar.set_ticklabels(ticklabels)
cbar.set_label('Frequency', rotation=270,loc='center', labelpad=15)
plt.savefig('chignolin_dist_3.jpeg', dpi=1000, bbox_inches = "tight")
plt.show()

# Saving trajectories
states_indices = []
maxi_train = np.max(pred_ord, axis= 1)
coor_train = np.zeros_like(pred_ord)
for i in range(output_size):
    coor_train = np.where(pred_ord[:,i]== maxi_train)[0]
    states_indices.append(coor_train)
trajec = md.load(traj_file, top=top)
print(trajec)
trajec_0 = trajec[list(states_indices[0])]
print(trajec_0)
trajec_0.save_pdb("trajec_0.pdb", force_overwrite=True)
trajec_1 = trajec[list(states_indices[1])]
print(trajec_1)
trajec_1.save_pdb("trajec_1.pdb", force_overwrite=True)
trajec_2 = trajec[list(states_indices[2])]
print(trajec_2)
trajec_2.save_pdb("trajec_2.pdb", force_overwrite=True)

DeepWEST.create_westpa_dir(traj_file=traj_file, top=top, indices=indices_vamp, shuffled_indices=shuff_indexes)
os.chdir(westpa_cwd)
DeepWEST.run_min_chignolin_westpa_dir(traj=traj_file, top=top, maxcyc = 200, cuda="unavailable")
print("Creating WESTPA Filetree...")
#DeepWEST.create_westpa_filetree()
DeepWEST.create_biased_westpa_filetree(state_indices = sorted_indices, num_states = output_size)
os.chdir(current_cwd)
