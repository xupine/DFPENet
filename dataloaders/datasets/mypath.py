class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'vaihingen':
            return 'D:/Segmentation/train/'
        elif dataset == 'landslide'
            return 'D:/Segmentation/landslide/train'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
