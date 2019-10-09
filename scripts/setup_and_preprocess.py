from Reco3D.lib import dataset

if __name__ == '__main__':
    dataset.setup_dir()
    dataset.prepare_dataset()
    dataset.preprocess_dataset()
    print ('done...')

