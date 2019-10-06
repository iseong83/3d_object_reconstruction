import os
from Reco3D.lib import dataset

if __name__ == '__main__':
    print ('re-preprocessing data')
    os.remove('./data_preprocessed/*.npy')
    print ('removed the preprocessed data')
    dataset.preprocess_dataset()
    print ('done...')

