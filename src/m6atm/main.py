import argparse
from m6atm.preprocess.m6atm import *
from m6atm.train.predict import *
from m6atm.train.run_model import *

__version__ = '1.0.0'

def main():
    
    ### main parser
    parser = argparse.ArgumentParser(prog = 'm6atm', description = 'm6ATM v%s'%(__version__))
    subparsers = parser.add_subparsers(help = 'modules', metavar = '{run, preprocess, predict, visualize}')
    parser.add_argument('-v', '--version', action = 'version', version = 'm6ATM %s'%(__version__))
    
    ### m6atm 'run'
    subparser1 = subparsers.add_parser('run', help = 'all-in-one command for quick m6A detection')
    
    req1 = subparser1.add_argument_group('Required')
    req1.add_argument('-f', '--fastq', dest = 'fastq', metavar = '\b', type = str, required = True, help = 'fastq directory generated by Guppy basecaller')
    req1.add_argument('-b', '--bam', dest = 'bam', metavar = '\b', type = str, required = True, help = 'path to bam file')
    req1.add_argument('-r', '--ref', dest = 'ref', metavar = '\b', type = str, required = True, help = 'path to reference transcriptome')
    req1.add_argument('-o', '--out', dest = 'out', metavar = '\b', type = str, required = True, help = 'output directory')
    
    opt1 = subparser1.add_argument_group('Optional')
    opt1.add_argument('-P', '--prefix', dest = 'prefix', metavar = '\b', type = str, default = 'default', help = 'output file prefix')
    opt1.add_argument('-N', '--processes', dest = 'n_proc', metavar = '\b', type = int, default = 1, help = 'number of processes (default: 1)')
    opt1.add_argument('-M', '--mem', dest = 'mem', metavar = '\b', type = str, help = 'max memory use per process (default: 10GB)', default = '10GB')
    opt1.add_argument('-p', '--port', dest = 'port', metavar = '\b', type = str, help = 'port for dask scheculer (default: 8788)', default = '8788')
    opt1.add_argument('-l', '--min_len', metavar = '\b', type = int, help = 'minimum read length (default: 500)', default = 500)
    opt1.add_argument('-L', '--max_len', metavar = '\b', type = int, help = 'maximum read length (default: 20000)', default = 20000)
    opt1.add_argument('-s', '--min_read', metavar = '\b', type = int, help = 'minimum read number at each site (default: 20)', default = 20)
    opt1.add_argument('-S', '--max_read', metavar = '\b', type = int, help = 'maximum read length at each site (default: 1000)', default = 1000)
    
    opt1.add_argument('-t', '--tx', dest = 'tx', metavar = '\b', type = str, help = 'transcript table from UCSC')
    opt1.add_argument('-R', '--ref_gn', dest = 'ref_gn', metavar = '\b', type = str, help = 'path to reference genome')
    opt1.add_argument('-T', '--thres', dest = 'thres', metavar = '\b', type = float, default = 0.9, help = 'probability threshold (default: 0.9)')
    opt1.add_argument('-x', '--device', dest = 'device', metavar = '\b', type = str, default = 'cuda:0', help = '<cuda:id> or <cpu> (default: cuda:0)')
    
    opt1.add_argument('-Q', '--mode', dest = 'mode', metavar = '\b', type = str, default = 'run', help = 'run/preprocess/predict')
    
    dev1 = subparser1.add_argument_group('Development')
    dev1.add_argument('--keep_file', help = 'keep intermediate files', default = False, action = 'store_true')
    
    subparser1.set_defaults(func = run)
    
    
    ### m6atm 'visualize'
    subparser2 = subparsers.add_parser('visualize', help = 'transcript-to-genome coordinate conversion')
    
    req2 = subparser2.add_argument_group('Required')
    req2.add_argument('-i', '--input', dest = 'input', metavar = '\b', type = str, required = True, help = 'results.csv file from m6ATM')
    req2.add_argument('-t', '--tx', dest = 'tx', metavar = '\b', type = str, required = True, help = 'transcript table from UCSC')
    req2.add_argument('-R', '--ref_gn', dest = 'ref_gn', metavar = '\b', type = str, required = True, help = 'path to reference genome')
    
    opt2 = subparser2.add_argument_group('Optional')
    opt2.add_argument('-o', '--out', dest = 'out', metavar = '\b', type = str, help = 'output directory')

    subparser2.set_defaults(func = visualize)
    

    args = parser.parse_args()
    args.func(args)