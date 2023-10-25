### Modules
# basic
import os, glob, h5py, random, itertools
import numpy as np
from Bio.Seq import Seq
from Bio import SeqUtils
from sklearn import preprocessing
from scipy.interpolate import interp1d
from collections import OrderedDict
from npy_append_array import NpyAppendArray

# dask
import dask
import dask.bag as db
from dask.distributed import Client, LocalCluster

# scripts
from .ReadClass import get_ref_dict
from .ResquiggleUtils import get_traceboundary, correct_cigar


### Transformation
def equalize_feature(feature_np, target_len, padding = False):

    feature_len = feature_np.shape[1]
    
    if feature_len == target_len:
        return np.array(feature_np, dtype ='float64')
    
    elif feature_len<target_len:
        if padding:
            feature_np = np.pad(np.array(feature_np, dtype ='float64'),
                                [(0, 0), (0, target_len-feature_len)], mode = 'constant', constant_values = 0)
            return feature_np
        else:
            feature_np_interpl = []
            for i in range(feature_np.shape[0]):
                interpl = interpolate(feature_np[i,:], target_len, kind = 'quadratic')
                feature_np_interpl.append(interpl.reshape((1, -1)))
            feature_np_interpl = np.concatenate(feature_np_interpl, axis = 0)
            
            return feature_np_interpl
    
    else:
        feature_np_interpl = []
        for i in range(feature_np.shape[0]):
            interpl = downsampling(feature_np[i,:], target_len)
            feature_np_interpl.append(interpl.reshape((1, -1)))
        feature_np_interpl = np.concatenate(feature_np_interpl, axis = 0)

        return feature_np_interpl
    
def rearrange_trace(trace_vector):
    # rearrangement
    vector_a = [x+y for x,y in zip(list(zip(*trace_vector))[0], list(zip(*trace_vector))[4])]
    vector_c = [x+y for x,y in zip(list(zip(*trace_vector))[1], list(zip(*trace_vector))[5])]
    vector_g = [x+y for x,y in zip(list(zip(*trace_vector))[2], list(zip(*trace_vector))[6])]
    vector_u = [x+y for x,y in zip(list(zip(*trace_vector))[3], list(zip(*trace_vector))[7])]
    
    result = [vector_a, vector_c, vector_g, vector_u]
    
    return result

def interpolate(vector, target_len, kind = 'quadratic'):
    
    # original x and y
    x_observed = []
    y_observed = []

    vector_len = len(vector)
    unit = target_len/vector_len
    for i in range(0, vector_len):
        x_observed.append(i * unit)
        y_observed.append(vector[i])

    # curve
    method = lambda x, y: interp1d(x, y, kind = kind)

    try:
        fitted_curve = method(x_observed, y_observed)
    except ValueError:
        print(x_observed, y_observed)

    # new x points
    x_latent = np.linspace(min(x_observed), max(x_observed), target_len)
    y_new = fitted_curve(x_latent)
    y_new = np.array(y_new, dtype ='float64')
    
    return y_new

def downsampling(vector, target_len):
    
    vector_len = len(vector)
    interpolated = interp1d(np.arange(vector_len), np.array(vector, dtype ='float64'), axis = 0, fill_value = 'extrapolate')
    downsampled = interpolated(np.linspace(0, vector_len, target_len))
    
    return downsampled


### Data collection
def get_read_h5(dset, read, info = 'basic'):

    ### read_info from hdf5
    read_id = read
    trace = dset['trace'][:]
    move = dset['move'][:]
    signal = dset['signal'][:]
    slength = dset['slength'][0]
    seq = dset['seq'].asstr()[0]
    ctg = dset['ctg'].asstr()[0]

    read_info = [read_id, signal, slength, trace, move, ctg, seq]


    if info == 'basic':

        return read_info

    else:

        if 'cigar' not in dset.keys(): # if not mapped

            return read_info, 'no_mapped'

        ### mapping_info from hdf5
        cigar = dset['cigar'].asstr()[0]
        mappedref = dset['mappedref'].asstr()[0]
        queryref = dset['queryref'].asstr()[0]
        r_st, r_en, q_st, q_en = dset['pos'][0], dset['pos'][1], dset['pos'][2], dset['pos'][3]

        read_mapping = [cigar, mappedref, queryref, r_st, r_en, q_st, q_en]

        if info == 'mapped':

            return read_info, read_mapping

        elif info == 'traceback':

            # if no traceback
            if 'traceback' not in dset.keys():
                return read_info, read_mapping, 'no_traceback'

            ### traceback_info from hdf5
            traceback = dset['traceback'][:]
            traceboundary, seq_len = get_traceboundary(traceback, trace)
            cigar_vb = dset['cigar_vb'].asstr()[0]
            
            read_traceback = [traceback, traceboundary, cigar_vb]

            return read_info, read_mapping, read_traceback

def get_interval_viterbi(pos_center, signal, trace, cigar, mappedref, r_st, traceboundary, s_span = 2, t_span = 2):
    
    # pos -> pos_lost -> ref_mapped pos
    pos_s_interval = list(range(pos_center-s_span-1, pos_center+s_span+1))
    pos_t_interval = list(range(pos_center-t_span-1, pos_center+t_span+1)) 

    pos_s_interval = [pos-r_st for pos in pos_s_interval]
    pos_t_interval = [pos-r_st for pos in pos_t_interval]

    if (min(pos_t_interval)<=0) or (max(pos_t_interval)>=len(mappedref)): # trace interval out of border
        return 'out'
    
    # ref_mapped pos -> trace boundary idx
    bpos_s_interval = [correct_cigar(pos, cigar) for pos in pos_s_interval]
    bpos_t_interval = [correct_cigar(pos, cigar) for pos in pos_t_interval]
    
    if (min(bpos_t_interval)<0) or (max(bpos_t_interval)>=len(traceboundary[1:])): # out of traceboundary
        return 'out'

    # trace boundary idx -> trace pos
    bidx_s_interval = [traceboundary[1:][bpos] for bpos in bpos_s_interval]
    bidx_t_interval = [traceboundary[1:][bpos] for bpos in bpos_t_interval]
    
    if max(bidx_t_interval)>=trace.shape[0]: # signal too short
        return 'out' 

    signal_interval = [min(bidx_s_interval)*10, max(bidx_s_interval)*10]
    trace_interval = [min(bidx_t_interval), max(bidx_t_interval)]

    signal_data = signal[signal_interval[0]:signal_interval[1]]
    trace_data = trace[trace_interval[0]:trace_interval[1],:]

    motif = [mappedref.replace('U', 'T')[n] for n in pos_t_interval]
    motif = ''.join(motif[1:])
    
    return [signal_data, trace_data, signal_interval, trace_interval, motif]

def get_motif_ref(ref_dict, pattern_list):
    
    ref_list = list(ref_dict.items())
    motif_dict = OrderedDict()

    for ref in ref_list:
        
        loc_list = []
        for pattern in pattern_list:
        
            locations = SeqUtils.nt_search(str(ref[1]).upper().replace('U', 'T'), Seq(pattern))
            locs = [i+2 for i in locations[1:]]
            loc_list.extend(locs)
            
        motif_dict[ref[0]] = loc_list

    return motif_dict

def ft_np_viterbi(h5_file, ref, pattern, out_dir, file_label, s_span = 2, t_span = 2, ft_len = 256, padding = False):
    
    ### file
    file_name = os.path.splitext(os.path.basename(h5_file))[0]
    tag_list = []
    
    ### motifs: get motif dict from whole transcriptome or specific sites
    ref_dict = get_ref_dict(ref)
    motif_dict = get_motif_ref(ref_dict, pattern)
    
    ### feature table (numpy array)
    with h5py.File(h5_file, 'a') as f1, NpyAppendArray(os.path.join(out_dir, 'data_'+file_label+'_'+file_name+'.npy')) as f2:
        
        for read in f1['m6ATM']:
            
            ### read info
            dset = f1['m6ATM'][read]
            read_info, read_mapping, read_traceback = get_read_h5(dset, read, info = 'traceback')
            if read_traceback == 'no_traceback': continue # if failed to get traceback

            read_id, signal, slength, trace, move, ctg, seq = read_info
            cigar, mappedref, queryref, r_st, r_en, q_st, q_en = read_mapping
            traceback, traceboundary, cigar_vb = read_traceback

            ### motif search
            pos_list = motif_dict[ctg]
            if len(pos_list) == 0: continue # if no matched

            ### data interval 
            for pos in pos_list:
                # signal/trace data
                data_interval = get_interval_viterbi(pos, signal, trace, cigar_vb, mappedref, r_st, traceboundary,
                                                     s_span = s_span, t_span = t_span)

                if data_interval != 'out':
                    if data_interval[0].shape[0]>0.1*ft_len:
                    
                        signal_data, trace_data, signal_interval, trace_interval, motif = data_interval

                        # data to np
                        ft_np = []

                        # signal 
                        ft_np.append(signal_data.reshape((1, -1)))

                        # trace extension
                        trace_acgt = rearrange_trace(trace_data)
                        for i in trace_acgt:
                            extended = np.array(list(itertools.chain.from_iterable([[t/255]*10 for t in i])))
                            ft_np.append(extended.reshape((1, -1)))

                        ft_np = np.concatenate(ft_np, axis = 0)
                        ft_interpl = equalize_feature(ft_np, ft_len)
                        

                        tag_list.append(np.array(read_id+'/'+ctg+'/'+str(pos)+'/'+motif).reshape(1))

                        ### save data
                        f2.append(ft_interpl.reshape(-1).astype('float64'))
    
    tag_list = np.concatenate(tag_list)
    np.save(os.path.join(out_dir, 'tag_'+file_label+'_'+file_name+'.npy'), tag_list)
                    
    return 0

def FtDask(files, temp_dir, pattern = ['KGACY'], ref = None, out_dir = None, file_label = 'default',s_span = 2, t_span = 2, 
           ft_len = 256, padding = False, npartitions = 96):
    

    ### dask.bag for fast5 files
    h5_bags = db.from_sequence(files, npartitions = npartitions)
    dask_out = h5_bags.map(ft_np_viterbi, ref = ref,  pattern = pattern, out_dir = temp_dir, file_label = file_label,
                           s_span = s_span, t_span = t_span, ft_len = ft_len, padding = padding).compute()
    
    ### merge files
    # data.npy
    data_file_list = sorted(glob.glob(os.path.join(temp_dir, 'data*')))
    with NpyAppendArray(os.path.join(out_dir, 'data_'+file_label+'.npy')) as f:
        for np_file in data_file_list:
            np_data = np.load(np_file, mmap_mode = "r")
            f.append(np_data)
            
    # tag.npy
    tag_file_list = sorted(glob.glob(os.path.join(temp_dir, 'tag*')))
    tag_data_merged = []
    for tag_file in tag_file_list:
        tag_data = np.load(tag_file)
        tag_data_merged.append(tag_data)
        
    tag_data_merged = np.concatenate(tag_data_merged)
    np.save(os.path.join(out_dir, 'tag_'+file_label+'.npy'), tag_data_merged)
       
    return dask_out

