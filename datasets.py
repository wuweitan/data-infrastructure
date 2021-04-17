class dataset_protComp_pair:
    def __init__(self, ds_name, download=False, root='./datasets/'):
        self.ds_name = ds_name
        self.root = root
        if download:
            self.download()

    def download(self):
        raise NotImplementedError

    def get_seqAndSmile(self):
        raise NotImplementedError



class dataset_deepaffinity(dataset_protComp_pair):
    def __init__(self, ds_name, download=False, root='./datasets/'):
        super(dataset_deepaffinity, self).__init__(ds_name, download, root)

    def download(self):
        pass

    def get_seqAndSmile(self, subset='ic50_train'): # {ic50, ec50, ki, kd} * {train, test, er, ion_channel, gpcr, tyrosine_kinase}
        pass


class dataset_deeprelations(dataset_protComp_pair):
    def __init__(self, ds_name, download=False, root='./datasets/'):
        super(dataset_deeprelations, self).__init__(ds_name, download, root)

    def download(self):
        pass

    def get_seqAndSmile(self, subset='train'): # train, val, test, unseen_protein, unseen_compound, unseen_both
        pass


class dataset_platinum(dataset_protComp_pair):
    def __init__(self, ds_name, download=False, root='./datasets/'):
        super(dataset_platinum, self).__init__(ds_name, download, root)

    def download(self):
        pass

    def get_seqAndSmile(self, subset='ki'): # ki, kd
        pass

