### Modules
# basic
import pysam, numba, h5py
from Bio.Seq import Seq
from numba.typed import List

# dask
import dask
import dask.bag as db
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar

from .ResquiggleUtils import *

def ViterbiRead(dset, read_id):
    ### from hdf5
    trace = dset['trace'][:]
    move = dset['move'][:]
    signal = dset['signal'][:]
    ctg = dset['ctg'].asstr()[0]
    cigar = dset['cigar'].asstr()[0]
    mappedref  = dset['mappedref'].asstr()[0]
    r_st, r_en, q_st, q_en = dset['pos'][0], dset['pos'][1], dset['pos'][2], dset['pos'][3]

    # potential_move
    potential_move = get_change_point(trace, move) 

    # convert mapped jit list
    potential_mvidx = move_to_index(potential_move, SEGMENT_ALL)
    basecaller_mvidx = move_to_index(potential_move, SEGMENT_FROM_BASECALLER)
    trace_compact = to_compact(trace, potential_mvidx, basecaller_mvidx)
    trace_compact_map = move_to_map(potential_move)

    ### m_range for range check
    ct_start = 0
    ct_end = get_near_pos(trace_compact_map, q_en)
    ct_len = len(trace_compact)

    trace_compact_exon = trace_compact[ct_start:ct_end]

    m_range = List()
    for n in range(q_en-q_st):
        l = correct_cigar(n, cigar) # take cigar into consideration 
        m = get_near_pos(trace_compact_map, q_st+l) - ct_start
        m_range.append(m)

    ### ViterbiTraceback
    mappedref = str(Seq(mappedref).back_transcribe())
    traceback = ViterbiTraceback(mappedref, trace_compact_exon, potential_mvidx, ct_start, m_range)
    
    traceboundary, seq_len_vb = get_traceboundary(traceback, trace)
    cigar_vb = to_cigar(traceback, seq_len_vb) 
    
    return traceback, cigar_vb

def ViterbiH5(file, len_threshold):
    
    f = h5py.File(file, 'a')
    
    for read in f['m6ATM']:
        dset = f['m6ATM'][read]
        ctg = dset['ctg'].asstr()[0]
        length = int(dset['length'][0])
        mapped_length = len(dset['mappedref'].asstr()[0])
        
        ### 'skip' if in the following condition
        # traceback already exists 
        if 'traceback' in dset.keys(): continue
    
        # read length not in the desired interval
        if (min(length, mapped_length)<len_threshold[0]) or (max(length, mapped_length)>len_threshold[1]): continue

        ### Viterbi
        traceback, cigar_vb = ViterbiRead(dset, read)
        
        ###  Store
        dset.create_dataset('traceback', data = traceback)
        dset.create_dataset('cigar_vb', shape = (1,), data = cigar_vb)
        
    return file

def VbDask(files, len_threshold, npartitions = 96):
    
    ### dask.bag for fast5 files
    with ProgressBar():
        h5_bags = db.from_sequence(files, npartitions = npartitions)
        dask_out = h5_bags.map(ViterbiH5, len_threshold = len_threshold).compute()
    
    return dask_out