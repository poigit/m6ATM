import os, glob, torch, shutil
import numpy as np, pandas as pd
from torchvision import transforms
from importlib_resources import files

import m6atm.preprocess.ReadClass as RC
import m6atm.train.ModelData as MD

def predict(args):
    
    # args
    data_dir = args.out if args.out else ''
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
    bag_class = MD.ATMbag(temp_dir, n_range = [min_read, max_read], processes = processes)
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
        
        _ = MD.to_bed(os.path.join(data_dir, 'results.csv'), tx_file, ref_gn, data_dir)
        
    # delete temp dir
    if not args.keep_file:
        shutil.rmtree(temp_dir, ignore_errors = True)
    
    return 0


def visualize(args):
    
    # args
    csv_table = args.input
    tx_file = args.tx
    ref_gn =  args.ref_gn
    
    data_dir = args.out if args.out else ''
        
    _ = MD.to_bed(csv_table, tx_file, ref_gn, data_dir)
    
    return 0