import os, sys, glob, time, argparse, shutil
from dask.distributed import Client, LocalCluster

import m6atm.preprocess.ReadClass as RC
import m6atm.preprocess.ResquiggleUtils as RU
import m6atm.preprocess.ResquiggleViterbi as RV
import m6atm.preprocess.FeatureTable as FT
import m6atm.preprocess.OtherUtils as OU

def preprocess(args):
    
    # args
    out_dir = args.out if args.out else ''
    fastq_dir = args.fastq
    bam_file = args.bam
    ref_file =  args.ref
    job = args.prefix
    n_workers = args.n_proc
    dask_mem = args.mem
    port = ':'+ args.port
    
    len_threshold = [args.min_len, args.max_len]
    

    # 0: config
    temp_dir = os.path.join(out_dir, 'temp')
    os.makedirs(temp_dir, exist_ok = True)

    fast5_path = os.path.join(fastq_dir, 'workspace/**/*.fast5')
    hdf5_path = os.path.join(temp_dir, '*.hdf5')
    
    f5_files = sorted(glob.glob(fast5_path, recursive = True))
    ref_dict = RC.get_ref_dict(ref_file)
    pattern_list = ['DRACH']
    ft_len = 256
    npartitions = 96

    logger1 = OU.create_log('job', out_dir, job, clean = False) # log file      
    
    
    # 1: alignment data
    with LocalCluster(n_workers = n_workers, threads_per_worker = 1, memory_limit = dask_mem, dashboard_address = port,
                      local_directory = temp_dir) as cluster, Client(cluster) as client:
    
        logger1.info('m6ATM - Data preprocessing %s'%(time.strftime('%D:%H:%M:%S')))
        logger1.info('Load mapping results ... %s'%(time.strftime('%D:%H:%M:%S')))
        mapped_df_list = RC.MapDask(bam_file, temp_dir, out_dir = out_dir) 
        logger1.info('Finished. %s'%(time.strftime('%D:%H:%M:%S')))
        
    # 2: data collection
    with LocalCluster(n_workers = n_workers, threads_per_worker = 1, memory_limit = dask_mem, dashboard_address = port,
                  local_directory = temp_dir) as cluster, Client(cluster) as client:
        
        logger1.info('Collect feature data ... %s'%(time.strftime('%D:%H:%M:%S')))
        _ = RC.H5Dask(files = f5_files, mapped_df_list = mapped_df_list, ref_path = ref_file, out_dir = temp_dir, norm = 'modifiedz')
        logger1.info('Finished. %s'%(time.strftime('%D:%H:%M:%S')))
    
    # 3: Resquiggling 
    with LocalCluster(n_workers = n_workers, threads_per_worker = 1, memory_limit = dask_mem, dashboard_address = port,
              local_directory = temp_dir) as cluster, Client(cluster) as client:
        
        logger1.info('Start Viterbi segmentation... %s'%(time.strftime('%D:%H:%M:%S')))
        h5_files = sorted(glob.glob(hdf5_path))
        vb_results = RV.VbDask(h5_files, len_threshold = len_threshold)
        logger1.info('Finished. %s'%(time.strftime('%D:%H:%M:%S')))
    
    # 4: Output table
    with LocalCluster(n_workers = n_workers, threads_per_worker = 1, memory_limit = dask_mem, dashboard_address = port,
              local_directory = temp_dir) as cluster, Client(cluster) as client:
        
        logger1.info('Process Viterbi results... %s'%(time.strftime('%D:%H:%M:%S')))
        out_list = FT.FtDask(h5_files, temp_dir, pattern = pattern_list, ref = ref_file, 
                             out_dir = out_dir, file_label = job, ft_len = ft_len, padding = False)
        logger1.info('Data preprocessing finished. %s'%(time.strftime('%D:%H:%M:%S')))

    # delete temp dir
    if not args.keep_file:
        
        shutil.rmtree(temp_dir, ignore_errors = True)

    return 0