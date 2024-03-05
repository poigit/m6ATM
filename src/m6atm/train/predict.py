import os, glob, torch, shutil
import numpy as np, pandas as pd
from torchvision import transforms
from importlib_resources import files

import m6atm.preprocess.ReadClass as RC
import m6atm.train.ModelData as MD

def predict(args):
    
    # args
    data_dir = args.out
    tx_file = args.tx
    ref_gn =  args.ref_gn
    processes = args.n_proc
    thres = args.thres
    min_read = args.min_read
    max_read = args.max_read
    
    # temp dir 
    temp_dir = os.path.join(data_dir, 'temp')
    os.makedirs(temp_dir, exist_ok = True)
    
    # to bag
    bag_class = MD.ATMbag(data_dir, n_range = [min_read, max_read], processes = processes)
    bag_class.to_bag(temp_dir)
    
    # prediction
    bag_data_list = sorted(glob.glob(os.path.join(temp_dir, 'bag_*.npy')))
    bag_meta_list = sorted(glob.glob(os.path.join(temp_dir, 'site_*.csv')))

    results = []
    for f1, f2 in zip(bag_data_list, bag_meta_list):

        ### bags
        bag_data = np.load(f1, allow_pickle = True)
        bag_meta = pd.read_csv(f2, index_col = 0)

        ### dataloader
        dataset = MD.WNBagloader(data = list(bag_data),
                                 transform = transforms.Compose([MD.ToTensor()]),
                                 site = bag_meta['site'],
                                 coverage = bag_meta['coverage'])
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = False)

        ### prediction
        dsmil_pth = files('m6atm.model').joinpath('dsmil_ivt.pth')
        classifier_pth = files('m6atm.model').joinpath('classifer_ivt.pth')
        result = MD.dsmil_pred(dsmil_pth, classifier_pth, dataloader, out_dir = temp_dir, thres = thres)
        results.append(result)

    results = pd.concat(results, axis = 0)
    results = results.reset_index(drop = True)

    ### save
    results.to_csv(os.path.join(data_dir, 'results.csv'))

    ### m6a bed
    if None not in [args.tx, args.ref_gn]:
        
        tx_df = pd.read_csv(tx_file, sep = '\t')
        tx_df['name'] = [i.split('.')[0] for i in tx_df['name']]
        ref_dict_gn = RC.get_ref_dict(ref_gn)

        results_m6a = results[results.m6a == 'yes']
        results_m6a_gn = MD.tx_to_gn(results_m6a, tx_df, ref_dict_gn)

        bed_table = results_m6a_gn.loc[:,['chrom', 'gn_pos', 'gn_pos_1', 'name2', 'ratio', 'strand']]
        bed_table.columns = ['chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand']

        bed_table.to_csv(os.path.join(data_dir, 'results.bed'), sep = '\t', index = None, header = None)
        
    else:
        
        results_m6a = results[results.m6a == 'yes'].copy()
        results_m6a['position2'] = [i+1 for i in results_m6a['position']]
        results_m6a['name'] = results_m6a['transcript']
        
        bed_table = results_m6a.loc[:,['transcript', 'position', 'position2', 'name', 'ratio']]
        bed_table.columns = ['chrom', 'chromStart', 'chromEnd', 'name', 'score']
                                    
        bed_table.to_csv(os.path.join(data_dir, 'results.bed'), sep = '\t', index = None, header = None)
        
    # delete temp dir
    if not args.keep_file:
        shutil.rmtree(temp_dir, ignore_errors = True)
    
    return 0