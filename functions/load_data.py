import torch
from torch_geometric.data import Data
import numpy as np
from glob import glob
import os

class MarielDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, reduced_joints=False, xy_centering=True, seq_len=128, predicted_timesteps=1, file_path="data/mariel_*.npy", overlap=True):
        'Initialization'
        self.file_path      = file_path
        self.seq_len        = seq_len
        self.overlap        = overlap
        self.reduced_joints = reduced_joints # use a meaningful subset of joints
        self.data           = load_data(pattern=file_path) 
        self.xy_centering   = xy_centering
        self.n_joints       = 53
        self.n_dim          = 3
        self.predicted_timesteps = predicted_timesteps
        
<<<<<<< HEAD
        if self.overlap == True:
            print("\nGenerating overlapping sequences...")
        else:
            print("\nGenerating non-overlapping sequences...")   
=======
        print("")
        
        if self.overlap == True:
            print("Generating overlapping sequences...")
        else:
            print("Generating non-overlapping sequences...")   
>>>>>>> d7f95637d5b500c545114bbdcf8305cb07c60e89
        
        if self.xy_centering == True: 
            print("Using (x,y)-centering...")
        else: 
            print("Not using (x,y)-centering...")
            
        if self.reduced_joints == True: 
            print("Reducing joints...")
        else:
            print("Using all joints...")
        
    def __len__(self):
        'Denotes the total number of samples'
        if self.xy_centering: 
            data = self.data[1] # choose index 1, for the (x,y)-centered phrases
        else: 
            data = self.data[0] # choose index 0, for data without (x,y)-centering
        
        if self.overlap == True:
            # number of overlapping phrases up until the final complete phrase
            return len(data)-self.seq_len 
        else:
             # number of complete non-overlapping phrases
            return int(len(data)/self.seq_len)

    def __getitem__(self, index):
        'Generates one sample of data'  
<<<<<<< HEAD
        edge_index, is_skeleton_edge, reduced_joint_indices = edges(n_joints=self.n_joints)
=======
        edge_index, is_skeleton_edge, reduced_joint_indices = edges(reduced_joints=self.reduced_joints, seq_len=self.seq_len)
>>>>>>> d7f95637d5b500c545114bbdcf8305cb07c60e89
        
        if self.xy_centering == True: 
            data = self.data[1] # choose index 1, for the (x,y)-centered phrases
        else: 
            data = self.data[0] # choose index 0, for data without (x,y)-centering
        
        if self.reduced_joints == True: 
            data = data[:,reduced_joint_indices,:] # reduce number of joints if desired
            
        if self.overlap == True:  
            # non-overlapping phrases
            sequence = data[index:index+self.seq_len]
            prediction_target = data[index+self.seq_len:index+self.seq_len+self.predicted_timesteps]
        else: 
            # overlapping phrases
            index = index*self.seq_len
            sequence = data[index:index+self.seq_len]
            prediction_target = data[index+self.seq_len:index+self.seq_len+self.predicted_timesteps]

        sequence = np.transpose(sequence, [1,0,2]) # put n_joints first
        sequence = sequence.reshape((data.shape[1],self.n_dim*self.seq_len)) # flatten n_dim*seq_len into one dimension (i.e. node feature)
        prediction_target = np.transpose(prediction_target, [1,0,2]) # put n_joints first
        prediction_target = prediction_target.reshape((data.shape[1],self.n_dim*self.predicted_timesteps)) # flatten n_dim*predicted_timesteps into one dimension (i.e. node feature) per node (data.shape[1] = number of nodes)

        # Convert to torch objects
        sequence = torch.Tensor(sequence)
        prediction_target = torch.Tensor(prediction_target)
        edge_attr = torch.Tensor(is_skeleton_edge)
        
        return Data(x=sequence, y=prediction_target, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)

def load_data(pattern="data/mariel_*.npy"):
   # load up the six datasets, performing some minimal preprocessing beforehand
    datasets = {}
    ds_all = []
    
    exclude_points = [26,53]
    point_mask = np.ones(55, dtype=bool)
    point_mask[exclude_points] = 0
    
    for f in sorted(glob(pattern)):
        ds_name = os.path.basename(f)[7:-4]
        ds = np.load(f).transpose((1,0,2))
        ds = ds[500:-500, point_mask]
        ds[:,:,2] *= -1
        datasets[ds_name] = ds
        ds_all.append(ds)

    ds_counts = np.array([ds.shape[0] for ds in ds_all])
    ds_offsets = np.zeros_like(ds_counts)
    ds_offsets[1:] = np.cumsum(ds_counts[:-1])

    ds_all = np.concatenate(ds_all)
<<<<<<< HEAD
#     print("Full data shape:", ds_all.shape)
=======
>>>>>>> d7f95637d5b500c545114bbdcf8305cb07c60e89
    print("Original numpy dataset contains {:,} timesteps of {} joints with {} dimensions each.".format(ds_all.shape[0], ds_all.shape[1], ds_all.shape[2]))

    low,hi = np.quantile(ds_all, [0.01,0.99], axis=(0,1))
    xy_min = min(low[:2])
    xy_max = max(hi[:2])
    xy_range = xy_max-xy_min
    ds_all[:,:,:2] -= xy_min
    ds_all *= 2/xy_range
    ds_all[:,:,:2] -= 1.0

    # it's also useful to have these datasets centered, i.e. with the x and y offsets
    # subtracted from each individual frame

    ds_all_centered = ds_all.copy()
    ds_all_centered[:,:,:2] -= ds_all_centered[:,:,:2].mean(axis=1,keepdims=True)

    datasets_centered = {}
    for ds in datasets:
        datasets[ds][:,:,:2] -= xy_min
        datasets[ds] *= 2/xy_range
        datasets[ds][:,:,:2] -= 1.0
        datasets_centered[ds] = datasets[ds].copy()
        datasets_centered[ds][:,:,:2] -= datasets[ds][:,:,:2].mean(axis=1,keepdims=True)

    low,hi = np.quantile(ds_all, [0.01,0.99], axis=(0,1))
    
    ### OPTIONAL RESCALING FOR DEBUGGING GNN
#     from sklearn.preprocessing import MinMaxScaler
#     scaler = MinMaxScaler()
#     ds_all_centered = scaler.fit_transform(ds_all_centered.reshape(-1, ds_all_centered.shape[-1])).reshape(ds_all_centered.shape)
        
    return ds_all, ds_all_centered, datasets, datasets_centered, ds_counts

<<<<<<< HEAD
def edges(n_joints = 53):
    all_edges = [(i,j) for i in range(n_joints) for j in range(n_joints)]
    all_edges_reversed = [(i,j) for i in range(n_joints) for j in range(n_joints)]
    edge_index = np.row_stack([all_edges,all_edges_reversed])

    ### PS: See http://www.cs.uu.nl/docs/vakken/mcanim/mocap-manual/site/img/markers.png for detailed marker definitions
    
=======
def edges(reduced_joints, seq_len):
    ### PS: See http://www.cs.uu.nl/docs/vakken/mcanim/mocap-manual/site/img/markers.png for detailed marker definitions
>>>>>>> d7f95637d5b500c545114bbdcf8305cb07c60e89
    point_labels=['ARIEL','C7','CLAV','LANK','LBHD','LBSH','LBWT','LELB','LFHD',
              'LFRM','LFSH','LFWT','LHEL','LIEL','LIHAND','LIWR','LKNE','LKNI',
              'LMT1','LMT5','LOHAND','LOWR','LSHN','LTHI','LTOE','LUPA','MBWT',
              'MFWT','RANK','RBHD','RBSH','RBWT','RELB','RFHD','RFRM','RFSH',
              'RFWT','RHEL','RIEL','RIHAND','RIWR','RKNE','RKNI','RMT1','RMT5',
              'ROHAND','ROWR','RSHN','RTHI','RTOE','RUPA','STRN','T10']

    ### Define a subset of joints if we want to train on fewer joints that still capture meaningful body movement:
    reduced_joint_names = ['ARIEL', 'CLAV', 'RFSH', 'LFSH', 'RIEL', 'LIEL', 'RIWR', 'LIWR','RKNE','LKNE','RTOE','LTOE','LHEL','RHEL','RFWT','LFWT','LBWT','RBWT']
    reduced_joint_indices = [point_labels.index(joint_name) for joint_name in reduced_joint_names]
    
<<<<<<< HEAD
=======
    all_edges = [(i,j) for i in range(53) for j in range(53)]
    all_edges_reversed = [(i,j) for i in range(53) for j in range(53)]
    
    if reduced_joints == True:
        reduced_edges = [(i,j) for i in reduced_joint_indices for j in reduced_joint_indices]
        reduced_edges_reversed = [(j,i) for i in reduced_joint_indices for j in reduced_joint_indices]
        edge_index = np.row_stack([reduced_edges,reduced_edges_reversed])
    else:
        edge_index = np.row_stack([all_edges,all_edges_reversed])
        
        
>>>>>>> d7f95637d5b500c545114bbdcf8305cb07c60e89
    skeleton_lines = [
    #     ( (start group), (end group) ),
        (('LHEL',), ('LTOE',)), # toe to heel
        (('RHEL',), ('RTOE',)),
        (('LMT1',), ('LMT5',)), # horizontal line across foot
        (('RMT1',), ('RMT5',)),   
        (('LHEL',), ('LMT1',)), # heel to sides of feet
        (('LHEL',), ('LMT5',)),
        (('RHEL',), ('RMT1',)),
        (('RHEL',), ('RMT5',)),
        (('LTOE',), ('LMT1',)), # toe to sides of feet
        (('LTOE',), ('LMT5',)),
        (('RTOE',), ('RMT1',)),
        (('RTOE',), ('RMT5',)),
        (('LKNE',), ('LHEL',)), # heel to knee
        (('RKNE',), ('RHEL',)),
        (('LFWT',), ('RBWT',)), # connect pelvis
        (('RFWT',), ('LBWT',)), 
        (('LFWT',), ('RFWT',)), 
        (('LBWT',), ('RBWT',)),
        (('LFWT',), ('LBWT',)), 
        (('RFWT',), ('RBWT',)), 
        (('LFWT',), ('LTHI',)), # pelvis to thighs
        (('RFWT',), ('RTHI',)), 
        (('LBWT',), ('LTHI',)), 
        (('RBWT',), ('RTHI',)), 
        (('LKNE',), ('LTHI',)), 
        (('RKNE',), ('RTHI',)), 
        (('CLAV',), ('LFSH',)), # clavicle to shoulders
        (('CLAV',), ('RFSH',)), 
        (('STRN',), ('LFSH',)), # sternum & T10 (back sternum) to shoulders
        (('STRN',), ('RFSH',)), 
        (('T10',), ('LFSH',)), 
        (('T10',), ('RFSH',)), 
        (('C7',), ('LBSH',)), # back clavicle to back shoulders
        (('C7',), ('RBSH',)), 
        (('LFSH',), ('LBSH',)), # front shoulders to back shoulders
        (('RFSH',), ('RBSH',)), 
        (('LFSH',), ('RBSH',)),
        (('RFSH',), ('LBSH',)),
        (('LFSH',), ('LUPA',),), # shoulders to upper arms
        (('RFSH',), ('RUPA',),), 
        (('LBSH',), ('LUPA',),), 
        (('RBSH',), ('RUPA',),), 
        (('LIWR',), ('LIHAND',),), # wrist to hand
        (('RIWR',), ('RIHAND',),),
        (('LOWR',), ('LOHAND',),), 
        (('ROWR',), ('ROHAND',),),
        (('LIWR',), ('LOWR',),), # across the wrist 
        (('RIWR',), ('ROWR',),), 
        (('LIHAND',), ('LOHAND',),), # across the palm 
        (('RIHAND',), ('ROHAND',),), 
        (('LFHD',), ('LBHD',)), # draw lines around circumference of the head
        (('LBHD',), ('RBHD',)),
        (('RBHD',), ('RFHD',)),
        (('RFHD',), ('LFHD',)),
        (('LFHD',), ('ARIEL',)), # connect circumference points to top of head
        (('LBHD',), ('ARIEL',)),
        (('RBHD',), ('ARIEL',)),
        (('RFHD',), ('ARIEL',)),
    ]

    skeleton_idxs = []
    for g1,g2 in skeleton_lines:
        entry = []
        entry.append([point_labels.index(l) for l in g1][0])
        entry.append([point_labels.index(l) for l in g2][0])
        skeleton_idxs.append(entry)
<<<<<<< HEAD

    is_skeleton_edge = []
    for edge in np.arange(2*n_joints**2): 
=======
        
    is_skeleton_edge = []      
    for edge in np.arange(edge_index.shape[0]): 
>>>>>>> d7f95637d5b500c545114bbdcf8305cb07c60e89
        if [edge_index[edge][0],edge_index[edge][1]] in skeleton_idxs: 
            is_skeleton_edge.append(torch.tensor(1.0))
        else:
            is_skeleton_edge.append(torch.tensor(0.0))
<<<<<<< HEAD
    
    return torch.tensor(edge_index, dtype=torch.long), is_skeleton_edge, reduced_joint_indices
=======

    is_skeleton_edge = np.array(is_skeleton_edge)
    copies = np.tile(is_skeleton_edge, (seq_len,1)) # create copies of the 1D array for every timestep
    skeleton_edges_over_time = torch.tensor(np.transpose(copies))
    
    if reduced_joints == True: 
        ### Need to remake these lists to include only nodes 0-18 now
        reduced_edges = [(i,j) for i in np.arange(len(reduced_joint_indices)) for j in np.arange(len(reduced_joint_indices))]
        reduced_edges_reversed = [(j,i) for i in np.arange(len(reduced_joint_indices)) for j in np.arange(len(reduced_joint_indices))]
        edge_index = np.row_stack([reduced_edges,reduced_edges_reversed])
    
    return torch.tensor(edge_index, dtype=torch.long), skeleton_edges_over_time, reduced_joint_indices
>>>>>>> d7f95637d5b500c545114bbdcf8305cb07c60e89
