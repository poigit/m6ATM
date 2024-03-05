### Modules
# basic
import os, h5py, pysam, glob
import numpy as np, pandas as pd
from Bio import SeqIO
from ont_fast5_api.fast5_interface import get_fast5_file
from sklearn import preprocessing
from collections import OrderedDict

# dask
import dask
import dask.bag as db
from dask.distributed import Client, LocalCluster

### Functions

def get_mapping_results(tx, bamfile):
    # get mapping results for each tx
    mapped_tx = []
    samfile = pysam.AlignmentFile(bamfile, 'rb')
    
    for read in samfile.fetch(tx):
        mapped_row = pd.DataFrame([[read.query_name, tx, read.reference_start, read.reference_end, 
                                    read.query_alignment_start, read.query_alignment_end, read.cigarstring, read.is_reverse]], 
                                  columns = ['id', 'ctg', 'r_st', 'r_en', 'q_st', 'q_en', 'cigar_str', 'is_reverse'] )
        mapped_tx.append(mapped_row)
    
    # if no result, skip
    if mapped_tx != []:
        mapped_tx = pd.concat(mapped_tx)
    else:
        mapped_tx = None
        
    return mapped_tx


def MapDask(bamfile, df_dir, n_reads = 10**5, out_dir = None, npartitions = 96):
    
    # temp df dir 
    file_list = sorted(glob.glob(df_dir+'/mapped_df_**.csv'))
    
    # tx
    tx_list = list(pysam.AlignmentFile(bamfile, 'rb').references)
    
    if len(file_list) == 0: # if files exist, do nothing
        ### generate mapped_df.csv

        ### dask.bag for reference transcripts
        tx_bags = db.from_sequence(tx_list, npartitions = npartitions)

        ### dask multi-processing
        mapped_df = tx_bags.map(get_mapping_results, bamfile = bamfile).compute()
        mapped_df = pd.concat(mapped_df)

        mapped_df = mapped_df.reset_index(drop = True)
        
        df_list = []
        for n, i in enumerate(range(0, mapped_df.shape[0], n_reads)):
            subset = mapped_df.iloc[i:i+n_reads,:]
            filename = os.path.join(df_dir, 'mapped_df_'+str(n)+'.csv')

            df_list.append(subset)
            subset.to_csv(filename)

    else:
        df_list = [pd.read_csv(file, index_col = 0) for file in file_list]
    
    return df_list


def get_ref_dict(ref_path):
    
    ref_dict = OrderedDict()
    for record in SeqIO.parse(ref_path, 'fasta'):
        ref_dict[record.id] = record.seq
        
    return ref_dict

def H5features(fast5_path, mapped_df, ref_path, out_dir = None, norm = False, z_constant = 0.6745):
    '''
    Save processed per-read features [read_id, fastq, trace, move, signal] to new HDF5 file
    # fast5_path: path to basecalled fast5
    # mapped_df : dataframe containig mapping info
    # norm      : 'minmax', 'modifiedz' or False
    '''
    with h5py.File(out_dir+'/'+os.path.basename(fast5_path).split('.')[0]+'.hdf5', 'a') as f:

        with get_fast5_file(fast5_path, mode = 'r') as f5:
            
            mapped_id = mapped_df.id.tolist()
            ref_dict = get_ref_dict(ref_path)
            tx_list = list(ref_dict.keys())
            
            for read in f5.get_reads():
                ### id 
                read_id = read.read_id  
                
                ### skip unused reads
                if read_id not in mapped_id: continue # skip if the read not mapped
                
                # mapped info
                mapped_row = mapped_df[mapped_df.id == read_id]
                ctg = mapped_row.ctg.tolist()[0]
                if ctg not in tx_list: continue  # skip if the ctg out of range
                
                pos = np.array([int(mapped_row.r_st), int(mapped_row.r_en),
                                int(mapped_row.q_st), int(mapped_row.q_en)]).reshape(-1)
                
                cigar = mapped_row.cigar_str.tolist()[0]
                if mapped_row.is_reverse.tolist()[0]:
                    strand = '-'
                else:
                    strand = '+'
                
                ### features
                # latest analysis
                basecall_run = read.get_latest_analysis('Basecall_1D')

                # seq: 5' to 3'
                fastq = read.get_analysis_dataset(basecall_run, 'BaseCalled_template/Fastq')
                if fastq is None: continue
                fastq = fastq.split('\n')[1]
                length = np.array(len(fastq)).reshape(1)

                # trace, move, signal : 3' to 5'
                trace = read.get_analysis_dataset(basecall_run, 'BaseCalled_template/Trace')
                move = read.get_analysis_dataset(basecall_run, 'BaseCalled_template/Move')
                signal = read.get_raw_data(scale = True) # scale to pA 
                slength = np.array(signal.shape[0]).reshape(1)
                
                trace = trace[::-1]
                move = move[::-1]

                # signal 
                signal = signal[len(signal)-10*len(trace):][::-1]  # 'trace':'signal data points' = 1:10 (defined by Guppy)
                
                if norm == 'minmax': # MinMaxScaler
                    signal = signal.reshape(-1, 1).astype('float16')
                    signal_norm = preprocessing.MinMaxScaler().fit_transform(signal) # fit signal value into 0 - 1 
                    signal_used = signal_norm.reshape(-1)
                    
                elif norm == 'modifiedz': # Modified z score
                    signal = signal.reshape(-1).astype('float16')
                    med = np.median(signal)
                    mad = np.median(np.abs(signal-med))
                    signal_used = z_constant*(signal-med)/mad
                    
                elif norm == 'mad':
                    median = np.median(signal)
                    mad = np.median(np.abs(signal - median))
                    scaled_data = ((signal - median) / mad)*128+128
                    signal_used = scaled_data.astype(np.uint8)
  
                else:
                    signal_used = signal.reshape(-1).astype('float16')
                
    
                # mapped sequence on reference
                ref_seq = ref_dict[ctg].transcribe()
                ref_mapped = str(ref_seq[pos[0]:pos[1]]).upper()
                ref_query = str(fastq[pos[2]:pos[3]]).upper()
                
                ### store in hdf5
                grp = f.create_group('/m6ATM/'+read_id)

                grp.create_dataset('trace' , data = trace)
                grp.create_dataset('move' , data = move)
                grp.create_dataset('signal' , data = signal_used)
                
                grp.create_dataset('length' , data = length)
                grp.create_dataset('slength' , data = slength)
                
                grp.create_dataset('pos' , data = pos)
                grp.create_dataset('ctg' , shape = (1,), data = ctg)
                grp.create_dataset('strand' , shape = (1,), data = strand)
                grp.create_dataset('cigar', shape = (1,) , data = cigar)
                
                grp.create_dataset('seq', shape = (1,)  , data = fastq)
                grp.create_dataset('mappedref', shape = (1,) , data = ref_mapped)
                grp.create_dataset('queryref', shape = (1,) , data = ref_query)
                

def H5Dask(files, mapped_df_list, ref_path, out_dir, norm = 'modifiedz', npartitions = 96):
    
    for df in mapped_df_list:
    
        ### dask.bag for fast5 files
        f5_bags = db.from_sequence(files, npartitions = npartitions)

        ### dask multi-processing
        f5_bags.map(H5features, mapped_df = df, ref_path = ref_path, out_dir = out_dir, norm = norm).compute()

    return 0
