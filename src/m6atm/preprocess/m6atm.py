import os, sys, glob, time, argparse, shutil
from dask.distributed import Client, LocalCluster

import m6atm.preprocess.ReadClass as RC
import m6atm.preprocess.ResquiggleUtils as RU
import m6atm.preprocess.ResquiggleViterbi as RV
import m6atm.preprocess.FeatureTable as FT
import m6atm.preprocess.OtherUtils as OU

__version__ = '0.0.1'

def preprocess(args):
    
    # args
    out_dir = args.out
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
    shutil.rmtree(temp_dir, ignore_errors = True)
    
    return 0


def main():
    
    ### main parser
    parser = argparse.ArgumentParser(prog = 'm6atm', description = 'm6ATM v0.0.1')
    subparsers = parser.add_subparsers(help = 'modules', metavar = '{preprocess, predict, visualize}')
    parser.add_argument('-v', '--version', action = 'version', version = 'm6ATM %s'%(__version__))
    
    ### m6atm 'preprocess'
    subparser1 = subparsers.add_parser('preprocess', help = 'data preprocessing for Nanopore DRS data')
    
    req1 = subparser1.add_argument_group('Required')
    req1.add_argument('-f', '--fastq', dest = 'fastq', metavar = '\b', type = str, required = True, help = 'fastq directory generated by Guppy basecaller')
    req1.add_argument('-b', '--bam', dest = 'bam', metavar = '\b', type = str, required = True, help = 'path to bam file')
    req1.add_argument('-r', '--ref', dest = 'ref', metavar = '\b', type = str, required = True, help = 'path to reference file')
    req1.add_argument('-o', '--out', dest = 'out', metavar = '\b', type = str, required = True, help = 'output directory')
    
    opt1 = subparser1.add_argument_group('Optional')
    opt1.add_argument('-P', '--prefix', dest = 'prefix', metavar = '\b', type = str, default = 'default', help = 'output file prefix')
    opt1.add_argument('-N', '--processes', dest = 'n_proc', metavar = '\b', type = int, default = 1, help = 'number of processes (default: 1)')
    opt1.add_argument('-m', '--mem', dest = 'mem', metavar = '\b', type = str, help = 'max memory use per process (default: 10GB)', default = '10GB')
    opt1.add_argument('-p', '--port', dest = 'port', metavar = '\b', type = str, help = 'port for dask scheculer (default: 8788)', default = '8788')
    opt1.add_argument('-l', '--min_len', metavar = '\b', type = int, help = 'minimum read length (default: 500)', default = 500)
    opt1.add_argument('-L', '--max_len', metavar = '\b', type = int, help = 'maximum read length (default: 20000)', default = 20000)
    
    subparser1.set_defaults(func = preprocess)
    
    args = parser.parse_args()
    args.func(args)
    