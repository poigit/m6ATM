# basic
import os, glob, gc, random, math, tsaug
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from collections import OrderedDict
from Bio.Seq import IUPACData
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from itertools import product

# pytorch
import torch
from torchvision import transforms

try:
    from .DSMIL import *
    from .WaveNet import *
except:
    from DSMIL import *
    from WaveNet import *

class ATMbag():
    
    def __init__(self, data_dir, n_range = [20, 1000], len_range = [500, 20000], data_size = 0, processes = 4):
        
        '''
        Args:
            data_dir: path to all processed data
            n_range: read coverage
            len_range: read length

            data_size: downsampling of processed data. 0 indicates no downsampling. 

            transform (callable): transform to be applied on a bag.
        '''

        self.data_dir = data_dir
        self.data = glob.glob(os.path.join(data_dir, '*_data.npy'))[0]
        self.label = glob.glob(os.path.join(data_dir, '*_label.npy'))[0]
        self.fname = os.path.basename(self.data).split('_data')[0]
        
        self.n_range = n_range
        self.len_range = len_range
        self.data_size = data_size
        self.processes = processes
        
    def load_np(self, k = 0):
        
        data = np.load(self.data, mmap_mode = 'r')
        label = np.load(self.label, allow_pickle = True).squeeze()
        
        ### meta df
        meta_list = list(map(lambda x: x.split('/'), label))
        meta_list = list(zip(*meta_list))

        id_list, ctg_list, pos_list, motif_list = meta_list[0], meta_list[1], meta_list[2], meta_list[3]
        site_list = [x+'/'+str(y)+'/'+z for x, y, z in zip(ctg_list, pos_list, motif_list)]
        meta_df = pd.DataFrame({'id': id_list, 'ctg': ctg_list, 'pos': pos_list, 'motif': motif_list, 'site': site_list}) 

        ### down sampling 
        if k>0:
            idx = sorted(random.sample(list(range(data.shape[0])), k = k))
            data = data[idx,:]
            meta_df = meta_df.iloc[idx,:]

        return data, meta_df    
    
        
    def _get_idx(self, table):
        
        ### collect all reads 
        # remove site with insufficient read coverage
        count_table = table.groupby(by = 'site', as_index = False)['id'].count()
        site_list = count_table[count_table.id >= self.n_range[0]].site.tolist()

        # idx_list
        table_s = table[table.site.isin(site_list)].copy()
        table_s['idx'] = table_s.index.tolist()
        
        table_grp = table_s.groupby('site', as_index = False).agg({'idx':lambda x: self._agg_idx(x, self.n_range[1]), 'id': 'count'})
        table_grp.columns = ['site', 'idx_list', 'coverage']

        idx_list = table_grp.idx_list.tolist()
        idx_list = [sorted(i) for i in idx_list]
        site_list = table_grp.site.tolist()
        coverage_list = table_grp.coverage.tolist()
        
        return idx_list, site_list, coverage_list
    
    def _agg_idx(self, x, n_max = None):
        
        idx_list = list(x)
        
        if n_max:
            if len(idx_list)>n_max:
                idx_list = random.sample(idx_list, n_max)
            
        return idx_list
    
    def to_bag(self, out_dir, batch_size = 5000, random_size = None, fold = 1, downsample = None, gt_site = None, gt_site_rev = None):
        
        ### groupby site
        data, meta = self.load_np(k = self.data_size)
        idx_list, site_list, coverage_list = self._get_idx(meta)
        
        if gt_site:
            
            selected = [site in gt_site for site in site_list]

            idx_list = [i for i, k in zip(idx_list, selected) if k]
            site_list = [i for i, k in zip(site_list, selected) if k]
            coverage_list = [i for i, k in zip(coverage_list, selected) if k]
            
        elif gt_site_rev:
            
            selected = [site not in gt_site_rev for site in site_list]

            idx_list = [i for i, k in zip(idx_list, selected) if k]
            site_list = [i for i, k in zip(site_list, selected) if k]
            coverage_list = [i for i, k in zip(coverage_list, selected) if k]

        if downsample:
            if len(idx_list)>downsample:
                
                k_list = sorted(random.sample(range(len(idx_list)), downsample))
                idx_list = [idx_list[i] for i in k_list]
                site_list = [site_list[i] for i in k_list]
                coverage_list = [coverage_list[i] for i in k_list]
        
        ### batch size
        for n, i in tqdm(enumerate(range(0, len(idx_list), batch_size)), total = ((len(idx_list)-1)//batch_size)+1):

            idx_batch = idx_list[i:i+batch_size]
            site_batch = site_list[i:i+batch_size]
            coverage_batch = coverage_list[i:i+batch_size]
            
            if fold>1:
                bag_data_list = []
                bag_meta_list = []
                for i in range(fold):
                    bag_data = Parallel(n_jobs = self.processes)(delayed(get_subset)(data, idx, k = random_size) for idx in tqdm(idx_batch, leave = False))
                    bag_meta = pd.DataFrame({'site': site_batch, 'coverage': coverage_batch})
                    
                    bag_data_list.extend(bag_data)
                    bag_meta_list.append(bag_meta)
                    
                bag_meta_list = pd.concat(bag_meta_list, axis = 0)
                
            else:
                bag_data_list = Parallel(n_jobs = self.processes)(delayed(get_subset)(data, idx, k = random_size) for idx in tqdm(idx_batch, leave = False))
                bag_meta_list = pd.DataFrame({'site': site_batch, 'coverage': coverage_batch})
                
            np.save(os.path.join(out_dir, 'bag_%s_%s.npy'%(self.fname, n)), np.array(bag_data_list, dtype = object))
            bag_meta_list.to_csv(os.path.join(out_dir, 'site_%s_%s.csv'%(self.fname, n)))

            gc.collect()
    
    def to_id(self):
        
        ### groupby site
        data, meta = self.load_np(k = self.data_size)
        idx_list, site_list, coverage_list = self._get_idx(meta)

        site_dict = OrderedDict(zip(site_list, idx_list))

        return data, site_dict
    

def get_subset(data, idx, k = None):
    
    if idx:
        if k is not None:
            if k[0]<len(idx):

                sample_size = random.randint(k[0], k[1])
                while sample_size>=len(idx):
                    sample_size = max(k[0], int(sample_size*0.5))

                idx = sorted(random.sample(idx, k = sample_size))
        
        bag = data[idx,:]
        bag = bag.reshape(bag.shape[0], bag.shape[1], 1)
        
    else:
        bag = None
        
    return bag


class WNBagloader():
    
    def __init__(self, data, label = None, transform = None, site = None, coverage = None, signal_only = False):
        
        '''
        Args:
            data: list of bags
            transform (callable): transform to be applied on a bag.
        '''
        self.data = data # list
        self.label = label # pct
        self.transform = transform
        self.site = site
        self.coverage = coverage
        self.signal = signal_only
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        data = self.data[idx].astype('float64')
        data = data.swapaxes(1, 2) # (n_queries, feature_dim, feature_len)
        
        if self.signal:
            data = data[:,:,0:256]
        
        if self.label:
            pct = self.label[idx]
            label = 1 if pct>0 else 0
        else:
            pct = -1
            label = -1
  
        if self.transform:
            data = self.transform(data)

        return data, label, pct


class WNReadloader():
    
    def __init__(self, data, label, transform = None, signal_only = False):
        
        '''
        Args:
            data: read-level numpy array (n_queries, feature_len) 
            label: 1 or 0
        '''
        
        self.data = data if not signal_only else data[:,0:256]
        self.label = label
        self.transform = transform
        
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        data, label = self.data[idx,:], self.label[idx]
        data = data.reshape(1, data.shape[0])
        
        if self.transform:
            data = self.transform(data)

        return data, label    
    
    
class ToTensor(object):
    def __init__(self, device = 'auto'):
        
        if device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        self.device = device
    
    def __call__(self, data):
        
        data = torch.as_tensor(data, device = self.device)

        return data

    
class Augmentation(object):
    def __call__(self, data):
        
        method = random.choice(['noise', 'convolve', 'drift', 'pool', 'wrap'])
        signal = data[:,:,:256] # (n_queries, feature_dim, feature_len) 
        trace = data[:,:,256:] # (n_queries, feature_dim, feature_len) 
        
        signal = self.signal_aug(signal, augmenter = 'noise')
        data_aug = np.concatenate([signal, trace], axis = 2)

        return data_aug
    
    def signal_aug(self, data, augmenter = 'noise'):
    
        if augmenter == 'noise':
            scale = round(random.uniform(0.04, 0.06), 2)
            data_aug = tsaug.AddNoise(scale = scale).augment(data)
        elif augmenter == 'convolve':
            size = random.randint(12, 18)
            data_aug = tsaug.Convolve(window = 'flattop', size = size).augment(data)
        elif augmenter == 'drift':
            drift = round(random.uniform(0.15, 0.25), 2)
            data_aug = tsaug.Drift(max_drift = drift, n_drift_points = 5).augment(data)
        elif augmenter == 'pool':
            size = random.randint(2, 5)
            data_aug = tsaug.Pool(size = size).augment(data)
        elif augmenter == 'wrap':
            ratio = round(random.uniform(1.1, 1.5), 1)
            data_aug = tsaug.TimeWarp(n_speed_change = 2, max_speed_ratio = ratio).augment(data)

        return data_aug    
    

def dsmil_pred(dsmil_file, classifier_file, dataloader, out_dir, thres = 0.9, device = 'auto'):
    
    ### model
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    dropout = 0.2
    
    ### DSMIL
    i_classifier = WaveNetModel(layers = 3, blocks = 2, input_channels = 1, kernel_size = 2, dropout = dropout, num_classes = 1)
    b_classifier = BClassifier(input_size = 1024, output_class = 1) # input_size is output shape of wavenet !! output_class?
    dsmil = DSMIL(i_classifier, b_classifier).to(device)
    dsmil.load_state_dict(torch.load(dsmil_file)) # load state
    
    ### Read classifier
    classifier = ReadClassifier(dropout = dropout).to(device)
    classifier.load_state_dict(torch.load(classifier_file)) # load state
    
    ### prediction
    dsmil.eval()
    classifier.eval()
    pred_list = []
    ratio_list = []
    for n, batch in tqdm(enumerate(dataloader), total = len(dataloader)):

        # forward
        data, _, _ = batch[0].squeeze(0).float().to(device), batch[1].float().to(device), batch[2].float().to(device)

        # dsmil
        feats, pred_ins = dsmil.i_classifier.encoder(data)
        pred_bag, A, B = dsmil.b_classifier(feats, pred_ins)

        # bag prediction
        pred = torch.sigmoid(pred_bag)
        pred_list.append(pred.cpu().float().item())
        
        # ratio estimation
        r_pred, r_prob, = classifier.forward(feats)
        ratio_list.append(r_pred.cpu().float().mean().item())
        
    ### table
    site_list = dataloader.dataset.site
    coverage_list = dataloader.dataset.coverage

    prob_list = [round(i, 3) for i in pred_list]
    mod_list = ['yes' if i>=thres else 'no' for i in prob_list]
    
    result_table = pd.DataFrame({'transcript': [site.split('/')[0].split('.')[0] for site in site_list],
                                 'position': [int(site.split('/')[1]) for site in site_list],                
                                 'motif': [site.split('/')[2] for site in site_list],
                                 'coverage': coverage_list, 
                                 'probability': prob_list,
                                 'm6a': mod_list,
                                 'ratio': ratio_list})      

    return result_table    
    

def plot_learning(train_data, test_data, n_epochs = 10, data_type = 'loss', out_dir = None, fig_name = None):
    
    ### main
    sns.set_theme(style = 'white') # theme
    tab_color = sns.color_palette() # color palette
    fig, ax = plt.subplots(figsize = (10, 6)) # figure size

    ### ax
    if data_type == 'loss':
        ax.set_ylabel('Loss', fontsize = 25)
        ax.set(xlim = (0.5, n_epochs+1))
        loc = 'upper right'
    elif data_type == 'accu':
        ax.set_ylabel('Accuracy', fontsize = 25)
        ax.set(xlim = (0.5, n_epochs+1), ylim = (0.4, 1))
        loc = 'upper left'
    
    ax.set_xlabel('Epochs', fontsize = 25)
    ax.tick_params(labelsize = 15)
    
    # plot
    ax.plot(range(1, n_epochs+1), train_data[0:n_epochs], color = tab_color[0], linewidth = 3)
    ax.plot(range(1, n_epochs+1), test_data[0:n_epochs], color = tab_color[1], linewidth = 3)
    ax.legend(labels = ['Train', 'Test'], fontsize = 16, loc = loc)
    
    fig.savefig(os.path.join(out_dir, fig_name+'.png'), bbox_inches='tight', dpi = 300)
    
    
    
    
def tx_to_gn(results, tx_df, ref_dict_gn):
    
    ### settings
    n_kmer = 5
    margin = int((n_kmer-1)*0.5) # 5-mer
    shift_range = range(-2, 3) # range to be fixed

    ### merge with tx_df
    results = results.merge(tx_df, how = 'left', left_on = 'transcript', right_on = 'name')
    results = results[results.chrom.isin(list(ref_dict_gn.keys()))]

    ### gn conversion
    results_gn = results.copy()
    results_gn['gn'] = results_gn.apply(get_gn_pos, ref_dict_gn = ref_dict_gn, margin = margin, axis = 1)
    results_gn['gn_pos'] = [int(i.split('_')[0]) for i in results_gn['gn']]
    results_gn['gn_motif'] = [i.split('_')[1] for i in results_gn['gn']]

    ### try to fix false_table with shifts
    results_true = results_gn.loc[lambda row: row['motif'] == row['gn_motif']]
    results_false = results_gn.loc[lambda row: row['motif'] != row['gn_motif']]

    results_fixed = try_shift(results_false, ref_dict_gn, shift_range = shift_range, margin = margin)
    if len(results_fixed)>0:
        results_gn = pd.concat([results_true, results_fixed], axis = 0)
    else:
        results_gn = results_true

    ### add info
    results_gn = results_gn.reset_index(drop = True)
    results_gn['gn_pos_1'] = [i+1 for i in results_gn['gn_pos']]
    results_gn['gn_site'] = [x+'_'+str(y) for x, y in zip(results_gn['chrom'], results_gn['gn_pos_1'])]

    results_gn = results_gn.groupby('gn_site', as_index = False).agg({'transcript': lambda x: ','.join(x),
                                                                      'position': lambda x: merge_int(x),
                                                                      'motif': 'first',
                                                                      'probability': 'mean',
                                                                      'ratio': 'mean',
                                                                      'm6a': lambda x: ','.join(x),
                                                                      'name2': 'first',
                                                                      'gn_pos': 'first',
                                                                      'gn_pos_1': 'first',
                                                                      'chrom': 'first', 
                                                                      'strand': 'first',
                                                                      'coverage': 'sum',
                                                                      'gn_motif' : 'count'})

    results_gn.rename(columns = {'gn_motif': 'tx_count'}, inplace = True)
    
    return results_gn
    
    
def get_gn_pos(row, ref_dict_gn, margin):
    
    chrom = row['chrom']
    strand = row['strand']
    pos = row['position'] # based on transcriptome: 0-start
    
    ### exon info 
    exons_start = [int(i) for i in row['exonStarts'].split(',')[:-1]]
    exons_end = [int(i) for i in row['exonEnds'].split(',')[:-1]]

    ### tx to gn conversion
    exon_len = 0
    if strand == '+':
        for x, y in zip(exons_start, exons_end):
            exon_len_last = exon_len
            exon_len += (y-x)
            if (pos+1)<=exon_len:
                gn_index = x+(pos-exon_len_last)

                kmer = ref_dict_gn[chrom][gn_index-margin:gn_index+margin+1].upper()
                kmer = ''.join(kmer)
                break

    if strand == '-':
        for x, y in zip(exons_start[::-1], exons_end[::-1]):
            exon_len_last = exon_len
            exon_len += (y-x)
            if (pos+1)<=exon_len:
                gn_index = y-(pos-exon_len_last)-1

                kmer = ref_dict_gn[chrom][gn_index-margin:gn_index+margin+1].reverse_complement().upper()
                kmer = ''.join(kmer)
                break
    
    result = str(gn_index)+'_'+kmer
    
    return result


def try_shift(false_table, ref_dict_gn, shift_range = range(-2,3), margin = 2):
    
    t_table = []
    for shift in shift_range:

        table = false_table.copy()
        table.loc[:,'gn_pos'] = [int(i+shift) for i in table.gn_pos.tolist()]
        table.loc[:,'gn_motif'] = [str(ref_dict_gn[chrom][int(pos)-margin:int(pos)+margin+1].upper()) for chrom, pos in zip(table.chrom, table.gn_pos)]

        # true or false
        t_idx = [x == y for x, y in zip(table['motif'], table['gn_motif'])]
        t_row = table.loc[t_idx,:]

        if t_row.shape[0]>0:
            t_table.append(t_row)

    if len(t_table)>0:
        t_table = pd.concat(t_table, axis = 0)

    return t_table
    
    
def merge_int(x):
    
    str_list = list(map(str, x))
    merged = ','.join(str_list)
    return merged

def extend_ambiguous(seq):
    
    d = IUPACData.ambiguous_dna_values
    return list(map(''.join, product(*map(d.get, seq)))) 
    