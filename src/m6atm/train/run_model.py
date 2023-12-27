from m6atm.preprocess.m6atm import *
from m6atm.train.predict import *

def run(args):
    
    # args
    mode = args.mode
    
    if mode == 'run':
        _ = preprocess(args)
        _ = predict(args)
        
    elif mode == 'preprocess':
        _ = preprocess(args)
    
    elif mode == 'predict':
        _ = predict(args)
        
    else: 
        raise Exception('Incorrect Mode Name: %'%(mode))
                                    
    return 0
                            
                                    
                                    
                                    
                                    
                                    
                                    
                                    