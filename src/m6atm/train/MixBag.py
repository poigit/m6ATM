# basic
import os, random
import numpy as np
from tqdm.auto import tqdm
from joblib import Parallel, delayed

try:
    from .ModelData import *
except:
    from ModelData import *

def get_mixed_loader(bag_data, label_data, size_data = None, site_data = None, split = True, signal_only = False):
    
    ### dataloader
    if split: # for training 
        dataset = WNBagloader(data = bag_data, label = label_data, site = site_data, coverage = size_data,
                              signal_only = signal_only, transform = transforms.Compose([ToTensor()]))
        
        train_size = int(0.7*len(dataset))
        test_size = len(dataset)-train_size
        dataset_train, dataset_test = torch.utils.data.random_split(dataset, [train_size, test_size])
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = 1, shuffle = True) 
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size = 1, shuffle = True)

        return [dataloader_train, dataloader_test]
    
    else: # for prediction
        dataset = WNBagloader(data = bag_data, label = label_data, site = site_data, coverage = size_data,
                              signal_only = signal_only, transform = transforms.Compose([ToTensor()]))
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = False)
        
        return dataloader

    
def load_mixed_bags(data_dir, prefix):
    
    bag_mixed_list = np.load(os.path.join(data_dir, prefix+'_bag.npy'), allow_pickle = True)
    pct_mixed_list = np.load(os.path.join(data_dir, prefix+'_label.npy')).tolist()

    size_data = os.path.join(data_dir, prefix+'_size.npy')
    site_data = os.path.join(data_dir, prefix+'_site.npy')
    
    if (os.path.isfile(size_data)) & (os.path.isfile(site_data)):
        
        size_mixed_list = np.load(size_data).tolist()
        site_mixed_list = np.load(site_data, allow_pickle = True).tolist()
    
        return bag_mixed_list, pct_mixed_list, size_mixed_list, site_mixed_list
    
    else:
        
        return bag_mixed_list, pct_mixed_list


def get_mixed_bags(data_dir1, data_dir2, out_dir, prefix = 'train', n_range = [20, None], bag_size = [20, 30], pct_range = [0.2, 1], n_bags = 3, processes = 4):
    
    '''
    Args:
        data_dir1: positive data
        data_dir2: negative data
        n_range: read pool size
        bag_size: number of reads in mixed bags
        pct_range: % of positive reads in mixed bags
        n_bags: number of mixed bags created at each site
    '''
    
    ### bag classese
    bag_p = ATMbag(data_dir1, n_range = n_range)
    bag_n = ATMbag(data_dir2, n_range = n_range)

    ### get all idx list
    data_p, site_dict_p = bag_p.to_id()
    data_n, site_dict_n = bag_n.to_id()

    set1 = set(list(site_dict_p.keys()))
    set2 = set(list(site_dict_n.keys()))
    site_list = sorted(list(set.intersection(set1, set2)))
    n_sites = len(site_list)
    
    ### parallel premixing
    results = Parallel(n_jobs = processes)(delayed(get_premixed)(site, data_p, data_n, site_dict_p, site_dict_n,
                                                                 bag_size = bag_size,
                                                                 n_bags = n_bags,
                                                                 pct_range = pct_range) for site in tqdm(site_list))
    
    results = list(zip(*results))
    mixed_list_p = sum(results[0], [])
    mixed_list_n = sum(results[1], [])
    pct_mixed_list = sum(results[2], [])
    size_mixed_list = sum(results[3], [])
    site_mixed_list = sum(results[4], [])
    
    ### get mixed bags
    bag_mixed_list = []
    for mixed in list(zip(mixed_list_p, mixed_list_n)):

        if any(i is None for i in mixed):
            bag = [i for i in mixed if i is not None][0]
        else:
            bag = np.concatenate(mixed, axis = 0)

        np.random.shuffle(bag) # shuffle rows
        bag_mixed_list.append(bag)
    
    
    print('%s bags mixed at %s sites'%(len(bag_mixed_list), n_sites))
    
    ### save bags
    np.save(os.path.join(out_dir, prefix+'_bag.npy'), np.array(bag_mixed_list, dtype = object))
    np.save(os.path.join(out_dir, prefix+'_label.npy'), np.array(pct_mixed_list))
    np.save(os.path.join(out_dir, prefix+'_size.npy'), np.array(size_mixed_list))
    np.save(os.path.join(out_dir, prefix+'_site.npy'), np.array(site_mixed_list, dtype = object))
    
    return bag_mixed_list, pct_mixed_list, size_mixed_list, site_mixed_list


def get_premixed(site, data_p, data_n, site_dict_p, site_dict_n, bag_size = [20, 30], n_bags = 3, pct_range = [0.2, 1]):
    
    idx_list_p, idx_list_n = site_dict_p[site], site_dict_n[site]
    pos_idx_list, neg_idx_list, pct_list, size_list = mix_bags(idx_list_p, idx_list_n, bag_size = bag_size, pct_range = pct_range, n_bags = n_bags)

    premixed_p = [get_subset(data_p, idx) for idx in pos_idx_list]
    premixed_n = [get_subset(data_n, idx) for idx in neg_idx_list]
    
    site_list = [site]*len(size_list)
        
    return premixed_p, premixed_n, pct_list, size_list, site_list


def mix_bags(idx_list_p, idx_list_n, bag_size = [20, 30], pct_range = [0.2, 1], n_bags = 3):
    
    ### positive and negative bags
    pos_idx_list = []
    neg_idx_list = []
    pct_list = []
    size_list = []
    for label in [1, 0]:
    
        i = 0
        while i<n_bags:
            # pct & size
            sample_size = random.randint(bag_size[0], bag_size[1])
            if sample_size>min(len(idx_list_p), len(idx_list_n)):
                sample_size = min(len(idx_list_p), len(idx_list_n))
            
            pct = round(random.uniform(pct_range[0], pct_range[1]), 2)

            # pos & neg size
            if label == 1:
                pos_size = int(sample_size*pct)
                neg_size = sample_size - pos_size
                
            if label == 0:
                pos_size = 0
                neg_size = sample_size
                pct = 0
                
            if pos_size == 0:
                pos_idx = None
                neg_idx = sorted(random.sample(idx_list_n, k = neg_size))
            elif neg_size == 0:
                pos_idx = sorted(random.sample(idx_list_p, k = pos_size))
                neg_idx = None
            else:
                pos_idx = sorted(random.sample(idx_list_p, k = pos_size))
                neg_idx = sorted(random.sample(idx_list_n, k = neg_size))

            pos_idx_list.append(pos_idx)
            neg_idx_list.append(neg_idx)
            pct_list.append(pct)
            size_list.append(sample_size)

            i+=1
        
    return pos_idx_list, neg_idx_list, pct_list, size_list
