# ### Modules
# basic
import os, glob, logging, pysam, h5py
import pandas as pd

# multi
import multiprocessing
from multiprocessing import Pool
from functools import partial

# scripts
from .ResquiggleUtils import get_traceboundary


### logger
def create_log(tag, path, job, clean = False):
    
    if clean == True:
        for f in glob.glob(os.path.join(path, '*.log')):
            os.remove(f)
        
    logger = logging.getLogger(tag)
    logger.setLevel(level=logging.INFO)
    logger.addHandler(logging.FileHandler(os.path.join(path, job+'_'+tag+'.log')))
    
    return logger

### Read data 
def _get_read(file, n, info = 'basic'):
    
    f = h5py.File(file, 'a')
    for idx, read in enumerate(f['m6ATM']):

        if idx!=n: continue

        ### read_info from hdf5
        dset = f['m6ATM'][read]
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

            
def get_relpos(cigar, pos):
    a = pysam.AlignedSegment()
    a.cigarstring = cigar
    refpos = 0
    relpos = 0
    
    for operator, cigarlen in a.cigar:

        if operator == 0:  # match
            if refpos + cigarlen > pos:
                return relpos + (pos-refpos)
            relpos = relpos + cigarlen
            refpos = refpos + cigarlen
        elif operator == 2:  # Del
            refpos = refpos + cigarlen
            if refpos > pos:
                return 'del'
        elif operator == 1 or operator == 3:  # Ins or N
            relpos = relpos + cigarlen

    return 0


### Count table 
def bam_to_count(bamfile, tx_df = None, ncores = 4):
    
    pysam_data = pysam.AlignmentFile(bamfile, "rb")
    txs = pysam_data.references # all txs in ref
    
    ### count_table
    ncores_avail = multiprocessing.cpu_count()
    if ncores > int(ncores_avail*0.8):
        ncores = int(ncores_avail*0.8)
    
    with Pool(ncores) as p:
        count_table = p.map(partial(get_counts_tx, bamfile = bamfile), txs)
        
    ### remove empty list
    count_table_filtered = []
    for i in count_table:
        if len(i)>0:
            count_table_filtered.extend(i)

    count_table_filtered = pd.concat(count_table_filtered)    
    count_table_filtered = count_table_filtered.groupby('tx', as_index = False).count().rename(columns = {'id': 'count'})
    
    ### transcript length 
    len_list = []
    for tx in count_table_filtered.tx:
        length = pysam_data.get_reference_length(tx)
        len_list.append(length)

    count_table_filtered['length'] = len_list
    
    ### TPM table
    # gene
    if tx_df is not None:
        count_table_filtered = count_table_filtered.merge(tx_df.loc[:,['name', 'name2']], left_on = 'tx', right_on = 'name')
        count_table_filtered = count_table_filtered.loc[:,['tx', 'count', 'length', 'name2']]
        count_table_filtered.columns = ['tx', 'count', 'length', 'gene']

    # RPK
    count_table_filtered['RPK'] = [(count/length)*1000 for count, length in zip(count_table_filtered['count'], count_table_filtered['length'])]

    # TPM
    tpm_f = count_table_filtered.RPK.sum()/10**6
    count_table_filtered['TPM'] = [RPK/tpm_f for RPK in count_table_filtered['RPK']]
        
    return count_table_filtered

def get_counts_tx(tx, bamfile):
    
    samfile = pysam.AlignmentFile(bamfile, "rb") # pysam

    hits = []
    for read in samfile.fetch(tx):
        hit = pd.DataFrame([[read.query_name, tx]], columns = ['id', 'tx'])
        hits.append(hit)
    
    return hits

