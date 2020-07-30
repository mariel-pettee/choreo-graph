import os
from glob import glob
from tqdm import tqdm
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import juggle_axes
from IPython.display import HTML
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = '/workdisk/choreo-graph/ffmpeg' # for using html5 video in Jupyter notebook
# print(matplotlib.animation.writers.list()) # check that ffmpeg is loaded. if it's not there, use .to_jshtml() instead of .to_html5_video().


def load_data(pattern="data/mariel_*.npy"):
   # load up the six datasets, performing some minimal preprocessing beforehand
    datasets = {}
    ds_all = []
    
    exclude_points = [26,53]
    point_mask = np.ones(55, dtype=bool)
    point_mask[exclude_points] = 0
    
    for f in sorted(glob(pattern)):
        ds_name = os.path.basename(f)[7:-4]
        print("loading:", ds_name)
        ds = np.load(f).transpose((1,0,2))
        ds = ds[500:-500, point_mask] # Exclude 2 joints that are averages, and cut out the first & last 500 frames (which sometimes are completely static)
        print("\t Shape:", ds.shape)
        ds[:,:,2] *= -1 # Invert z direction
#         print("\t Min:", np.min(ds,axis=(0,1)))
#         print("\t Max:", np.max(ds, axis=(0,1)))
        datasets[ds_name] = ds
        ds_all.append(ds)

    ds_counts = np.array([ds.shape[0] for ds in ds_all])
    ds_offsets = np.zeros_like(ds_counts)
    ds_offsets[1:] = np.cumsum(ds_counts[:-1]) # starting timesteps of each separate capture session

    ds_all = np.concatenate(ds_all)
    print("Full data shape:", ds_all.shape)

    low,hi = np.quantile(ds_all, [0,1], axis=(0,1))
    xy_min = min(low[:2])
    xy_max = max(hi[:2])
    xy_range = xy_max-xy_min
    ds_all[:,:,:2] -= xy_min # subtract the minimum from the x and y components
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

    print("Full Data Min:", np.min(ds_all, axis=(0,1)))
    print("Full Data Max:", np.max(ds_all, axis=(0,1)))    
    print("Full (Centered) Data Min:", np.min(ds_all_centered, axis=(0,1)))
    print("Full (Centered) Data Max:", np.max(ds_all_centered, axis=(0,1)))
    return ds_all, ds_all_centered, datasets, datasets_centered, ds_counts


# these are the ordered label names of the 53 vertices
# (after the Labeling/SolvingHips points have been excised)
point_labels = ['ARIEL', 'C7',
          'CLAV', 'LANK',
          'LBHD', 'LBSH',
          'LBWT', 'LELB',
          'LFHD', 'LFRM',
          'LFSH', 'LFWT',
          'LHEL', 'LIEL',
          'LIHAND', 'LIWR',
          'LKNE', 'LKNI',
          'LMT1', 'LMT5',
          'LOHAND', 'LOWR',
          'LSHN', 'LTHI',
          'LTOE', 'LUPA',
          #'LabelingHips',
          'MBWT',
          'MFWT', 'RANK',
          'RBHD', 'RBSH',
          'RBWT', 'RELB',
          'RFHD', 'RFRM',
          'RFSH', 'RFWT',
          'RHEL', 'RIEL',
          'RIHAND', 'RIWR',
          'RKNE', 'RKNI',
          'RMT1', 'RMT5',
          'ROHAND', 'ROWR',
          'RSHN', 'RTHI',
          'RTOE', 'RUPA',
          'STRN',
          #'SolvingHips',
          'T10']

# This array defines the points between which skeletal lines should
# be drawn. Each segment is defined as a line between a group of one
# or more named points -- the line will be drawn at the average position
# of the points in the group
# PS: See http://www.cs.uu.nl/docs/vakken/mcanim/mocap-manual/site/img/markers.png for detailed marker definitions
# skeleton_lines = [
# #     ( (start group), (end group) ),
#     (('LHEL',), ('LTOE',)), # toe to heel
#     (('RHEL',), ('RTOE',)),
#     (('LKNE','LKNI'), ('LHEL',)), # heel to knee
#     (('RKNE','RKNI'), ('RHEL',)),
#     (('LKNE','LKNI'), ('LFWT','RFWT','LBWT','RBWT')), # knee to "navel"
#     (('RKNE','RKNI'), ('LFWT','RFWT','LBWT','RBWT')),
#     (('LFWT','RFWT','LBWT','RBWT'), ('STRN','T10',)), # "navel" to chest
#     (('STRN','T10',), ('CLAV','C7',)), # chest to neck
#     (('CLAV','C7',), ('LFSH','LBSH',),), # neck to shoulders
#     (('CLAV','C7',), ('RFSH','RBSH',),),
#     (('LFSH','LBSH',), ('LELB', 'LIEL',),), # shoulders to elbows
#     (('RFSH','RBSH',), ('RELB', 'RIEL',),),
#     (('LELB', 'LIEL',), ('LOWR','LIWR',),), # elbows to wrist
#     (('RELB', 'RIEL',), ('ROWR','RIWR',),),
#     (('LFHD',), ('LBHD',)), # draw lines around circumference of the head
#     (('LBHD',), ('RBHD',)),
#     (('RBHD',), ('RFHD',)),
#     (('RFHD',), ('LFHD',)),
#     (('LFHD',), ('ARIEL',)), # connect circumference points to top of head
#     (('LBHD',), ('ARIEL',)),
#     (('RBHD',), ('ARIEL',)),
#     (('RFHD',), ('ARIEL',)),
# ]

### Use these instead for Anna & Jeannie film animations:
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
    (('LKNE','LKNI'), ('LHEL',)), # heel to knee
    (('RKNE','RKNI'), ('RHEL',)),
    (('LKNE',), ('LHEL',)), # heel to knee
    (('RKNE',), ('RHEL',)),
#     (('LKNE','LKNI'), ('LFWT','RFWT','LBWT','RBWT')), # knee to "navel"
#     (('RKNE','RKNI'), ('LFWT','RFWT','LBWT','RBWT')),
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
    (('LKNE','LKNI'), ('LTHI',)), # thighs to knees
    (('RKNE','RKNI'), ('RTHI',)), 
    (('LKNE',), ('LTHI',)), 
    (('RKNE',), ('RTHI',)), 
    (('LFWT','RFWT','LBWT','RBWT'), ('STRN','T10',)), # "navel" to chest
    (('STRN','T10',), ('CLAV','C7',)), # chest to neck
#     (('CLAV',), ('C7',)), # clavicle through to the back of chest
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
    (('CLAV','C7',), ('LFSH','LBSH',),), # neck to shoulders
    (('CLAV','C7',), ('RFSH','RBSH',),),
    (('LFSH','LBSH',), ('LELB', 'LIEL',),), # shoulders to elbows
    (('RFSH','RBSH',), ('RELB', 'RIEL',),),
    (('LFSH',), ('LUPA',),), # shoulders to upper arms
    (('RFSH',), ('RUPA',),), 
    (('LBSH',), ('LUPA',),), 
    (('RBSH',), ('RUPA',),), 
    (('LELB', 'LIEL',), ('LUPA',),), # upper arms to elbows
    (('RELB', 'RIEL',), ('RUPA',),),
#     (('LELB', 'LIEL',), ('LOWR','LIWR',),), # elbows to wrist
#     (('RELB', 'RIEL',), ('ROWR','RIWR',),),
    (('LELB', 'LIEL',), ('LIWR',),), 
    (('RELB', 'RIEL',), ('RIWR',),),
    (('LELB', 'LIEL',), ('LOWR',),), 
    (('RELB', 'RIEL',), ('ROWR',),),
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


# Normal, connected skeleton:
skeleton_idxs = []
for g1,g2 in skeleton_lines:
    entry = []
    entry.append([point_labels.index(l) for l in g1])
    entry.append([point_labels.index(l) for l in g2])
    skeleton_idxs.append(entry)

# Cloud of every point connected:
cloud_idxs = []
for i in range(53):
    for j in range(53):
        entry = []
        entry.append([i])
        entry.append([j])
        cloud_idxs.append(entry)

all_idxs = skeleton_idxs+cloud_idxs

# calculate the coordinates for the lines
def get_line_segments(seq, zcolor=None, cmap=None, cloud=False, edge_types=None, edge_class=None):
    xline = np.zeros((seq.shape[0],len(all_idxs),3,2))
    if cmap:
        colors = np.zeros((len(all_idxs), 4))
    for edge,(joint1,joint2) in enumerate(all_idxs):
        xline[:,edge,:,0] = np.mean(seq[:,joint1], axis=1)
        xline[:,edge,:,1] = np.mean(seq[:,joint2], axis=1)    
        if cmap:
            if edge_types is not None:
                if edge > len(skeleton_idxs)-1: # cloud edges
                    if edge_types[edge-len(skeleton_idxs), edge_class] == 1:
                        colors[edge] = cmap(1)
                    else:
                        colors[edge] = cmap(0)
            else:
                colors[edge] = cmap(0)
    if cmap:
        return xline, colors
    else:
        return xline
    
# put line segments on the given axis, with given colors
def put_lines(ax, segments, color=None, lw=2.5, alpha=None, cloud=False, edge_types=None, edge_class=None):
    lines = []
    ### Main skeleton
    for i in tqdm(range(len(skeleton_idxs)), desc="Skeleton lines"):
        if isinstance(color, (list,tuple,np.ndarray)):
            c = color[i]
        else:
            c = color
            
        ### THESE LINES PLOT THE MAIN SKELETON
#         l = ax.plot(np.linspace(segments[i,0,0],segments[i,0,1],2),
#                 np.linspace(segments[i,1,0],segments[i,1,1],2),
#                 np.linspace(segments[i,2,0],segments[i,2,1],2),
#                 color=c,
#                 alpha=alpha,
#                 lw=lw)[0]
#         lines.append(l)
    
    if cloud:
        ### Cloud of all-connected joints
        for i in tqdm(range(len(skeleton_idxs),len(all_idxs)), desc="Cloud lines"):
            if isinstance(color, (list,tuple,np.ndarray)):
                c = color[i]
            else:
                c = color
            ### plot or don't plot lines based on edge class
            if edge_types is not None and edge_class is not None:
                custom_colors = ["deeppink","black","red","blue"]
                if edge_types[i-len(skeleton_idxs)][edge_class] == 1:
                    l = ax.plot(np.linspace(segments[i,0,0],segments[i,0,1],2),
                            np.linspace(segments[i,1,0],segments[i,1,1],2),
                            np.linspace(segments[i,2,0],segments[i,2,1],2),
#                                 color=c,
                            color=custom_colors[edge_class],
                            alpha=0.1,
                            lw=lw)[0]
                    lines.append(l)
    return lines

# animate a video of the stick figure.
# `ghost` may be a second sequence, which will be superimposed
# on the primary sequence.
# If ghost_shift is given, the primary and ghost sequence will be separated laterally
# by that amount.
# `zcolor` may be an N-length array, where N is the number of vertices in seq, and will
# be used to color the vertices. Typically this is set to the avg. z-value of each vtx.
def animate_stick(seq, ghost=None, ghost_shift=0, edge_types=None, edge_class=None, figsize=None, zcolor=None, pointer=None, ax_lims=(-0.4,0.4), speed=45,
                  dot_size=20, dot_alpha=0.5, lw=2.5, cmap='cool_r', pointer_color='black', cloud=False):
    if zcolor is None:
        zcolor = np.zeros(seq.shape[1])
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    
    # The following lines eliminate background lines/axes:
    ax.axis('off')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    
    # set figure background opacity (alpha) to 0:
    fig.patch.set_alpha(0.)
    
    if ghost_shift and ghost is not None:
        seq = seq.copy()
        ghost = ghost.copy()
        seq[:,:,0] -= ghost_shift
        ghost[:,:,0] += ghost_shift
    
    cm = matplotlib.cm.get_cmap(cmap)
    
    if edge_types is not None:
        num_connections = np.zeros((53))
        for edge,(joint1,joint2) in enumerate(all_idxs):
            if edge > len(skeleton_idxs) and edge_types[edge - len(skeleton_idxs),edge_class] == 1:
                num_connections[joint1] += 1
                num_connections[joint2] += 1
        num_connections = num_connections/np.sum(num_connections) # normalize so colormap can use values between 0 and 1
#         print(num_connections)
        dot_color = [cm(num_connections[joint]) for joint in range(53)]
        pts = ax.scatter(seq[0,:,0],seq[0,:,1],seq[0,:,2], c=dot_color, s=dot_size, cmap=cm, alpha=dot_alpha)

    else:
        dot_color = "black"
        pts = ax.scatter(seq[0,:,0],seq[0,:,1],seq[0,:,2], c=dot_color, s=dot_size, cmap=cm, alpha=dot_alpha)
    
    ghost_color = 'blue'

    if ghost is not None:
        pts_g = ax.scatter(ghost[0,:,0],ghost[0,:,1],ghost[0,:,2], c=ghost_color, s=dot_size, alpha=dot_alpha)
    
    if ax_lims:
        ax.set_xlim(*ax_lims)
        ax.set_ylim(*ax_lims)
        ax.set_zlim(0,ax_lims[1]-ax_lims[0])
    plt.close(fig)
    xline, colors = get_line_segments(seq, zcolor, cm, edge_types=edge_types, edge_class=edge_class)
    lines = put_lines(ax, xline[0], color=colors, lw=lw, alpha=0.9, cloud=cloud, edge_types=edge_types, edge_class=edge_class)
    
    if ghost is not None:
        xline_g = get_line_segments(ghost)
        lines_g = put_lines(ax, xline_g[0], ghost_color, lw=lw, alpha=1.0, cloud=cloud)
    
    if pointer is not None:
        vR = 0.15
        dX,dY = vR*np.cos(pointer), vR*np.sin(pointer)
        zidx = point_labels.index('CLAV')
        X = seq[:,zidx,0]
        Y = seq[:,zidx,1]
        Z = seq[:,zidx,2]
        quiv = ax.quiver(X[0],Y[0],Z[0],dX[0],dY[0],0, color=pointer_color)
        ax.quiv = quiv
    
    def update(t):
        pts._offsets3d = juggle_axes(seq[t,:,0], seq[t,:,1], seq[t,:,2], 'z')
        for i,l in enumerate(lines):
            l.set_data(xline[t,i,:2])
            l.set_3d_properties(xline[t,i,2])
        
        if ghost is not None:
            pts_g._offsets3d = juggle_axes(ghost[t,:,0], ghost[t,:,1], ghost[t,:,2], 'z')
            for i,l in enumerate(lines_g):
                l.set_data(xline_g[t,i,:2])
                l.set_3d_properties(xline_g[t,i,2])
        
        if pointer is not None:
            ax.quiv.remove()
            ax.quiv = ax.quiver(X[t],Y[t],Z[t],dX[t],dY[t],0,color=pointer_color)
     
    return animation.FuncAnimation(
        fig,
        update,
        len(seq),
        interval=speed,
        blit=False,
   )
    
# draw a "comic strip" style rendering of the given sequence of poses
def draw_comic(frames, angles=None, figsize=None, window_size=0.45, dot_size=20, lw=2.5, zcolor=None,cmap='cool_r'):
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    ax.view_init(30, 0)
    shift_size=window_size
    
    ax.set_xlim(-window_size,window_size)
    ax.set_ylim(-window_size,len(frames)*window_size)
    ax.set_zlim(-0.1,0.6)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    
    cm = matplotlib.cm.get_cmap(cmap)
    
    if angles is not None:
        vR = 0.15
        zidx = point_labels.index("CLAV")
        X = frames[:,zidx,0]
        Y = frames[:,zidx,1]
        dX,dY = vR*np.cos(angles), vR*np.sin(angles)
        Z = frames[:,zidx,2]
        #Z = frames[:,2,2]
 
    
    for iframe,frame in enumerate(frames):
        ax.scatter(frame[:,0],
                       frame[:,1]+iframe*shift_size,
                       frame[:,2],
                       alpha=0.3,
                       c=zcolor,
                       cmap=cm,
                       s=dot_size,
                       depthshade=True)
        
        if angles is not None:
            ax.quiver(X[iframe],iframe*shift_size+Y[iframe],Z[iframe],dX[iframe],dY[iframe],0, color='black')
        
        for i,(g1,g2) in enumerate(skeleton_lines):
            g1_idx = [point_labels.index(l) for l in g1]
            g2_idx = [point_labels.index(l) for l in g2]

            if zcolor is not None:
                color = cm(0.5*(zcolor[g1_idx].mean() + zcolor[g2_idx].mean()))
            else:
                color = None

            x1 = np.mean(frame[g1_idx],axis=0)
            x2 = np.mean(frame[g2_idx],axis=0)
            
            ax.plot(np.linspace(x1[0],x2[0],10),
                    np.linspace(x1[1],x2[1],10)+iframe*shift_size,
                    np.linspace(x1[2],x2[2],10),
                    color=color,
                    lw=lw)

# Rotate a (?,...,?,3) tensor about the z-axis
def rotate(X, theta):
    c,s = np.cos(theta), np.sin(theta)
    R = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1.]
        ])
    return np.dot(X, R)
