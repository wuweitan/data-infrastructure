import torch
import numpy as np

class SCOPe_graphs(torch.utils.data.Dataset):
    ''' 
    Sample graphs and nodes in graph
    '''
    def __init__(self, path, normalize=True, seq_mask_dim = 21, max_num_nodes=60, max_SSE=None, DGL=False):
        self.adj_all = []
        self.len_all = []
        self.feature_all = []
        self.label_all = []
        self.seq = []
        self.seq_mask_all = []
        self.channel_num = len(G_list[0][1])
        self.DGL = DGL

        if max_num_nodes == 0:
            self.max_num_nodes = max([G[1][0].number_of_nodes() for G in G_list])
        else:
            self.max_num_nodes = max_num_nodes

        ### max element sequence length (SZ add)
        self.max_ele_seq_lengths = max([max([se_eb.shape[0] for se_eb in G[0]]) for G in G_list])
        print('max sequence element length: %d'%self.max_ele_seq_lengths)
        if max_SSE:
            self.padding_length = max_SSE
        else:
            self.padding_length = self.max_ele_seq_lengths
        print('Padding to the length of %d.'%self.padding_length)

        self.feat_dim = node_dict(G_list[0][1][0])[0]['feat'].shape[0]

        for sample in G_list:
            ### adjacency matrices ###
            G_all = sample[1]

            node_num = G_all[0].number_of_nodes()
            if node_num > self.max_num_nodes:
                continue # ignore the graphs whose amount of nodes are larger than max_nodes

            adj_tensor = []
            for G in G_all:
                adj = np.array(nx.to_numpy_matrix(G))
                if normalize and adj.shape[-1] > 1:
                    sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj + np.eye(adj.shape[-1]), axis=-1, dtype=float).squeeze())) #
                    if DGL:
                        adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)
                    else:
                        adj = np.matmul(np.matmul(sqrt_deg, adj + np.eye(adj.shape[-1])), sqrt_deg)
                elif DGL:
                    adj += np.eye(adj.shape[-1])
                adj_tensor.append(adj)
            adj_tensor = np.array(adj_tensor)
            if DGL:
                adj_tensor = Tensor_to_DGL(adj_tensor)

            G = G_all[0]
            ### node features ###
            # feat matrix: max_num_nodes x feat_dim
            f = np.zeros((self.channel_num, self.max_num_nodes, self.feat_dim), dtype=float)
            for i,u in enumerate(G.nodes()):
                f[:,i,:] = node_dict(G)[u]['feat']

            ### sequence ###
            g_seq = sample[0]
            node_num = len(g_seq)
            seq_emb = np.zeros((self.max_num_nodes, self.padding_length, seq_mask_dim), dtype=float)
            seq_mask = np.zeros((self.max_num_nodes,1), dtype=float)

            for i in range(node_num):
                seq_mask[i] = 1
                seq_len = g_seq[i].shape[0]
                seq_emb[i,:seq_len,:20] = g_seq[i]
                seq_emb[i,seq_len:,20] = 1

            self.seq.append(seq_emb)
            self.seq_mask_all.append(seq_mask)

            ### add the info into the lists
            self.adj_all.append(adj_tensor)
            self.feature_all.append(f)
            self.seq.append(seq_emb)
            self.seq_mask_all.append(seq_mask)
            self.len_all.append(node_num)
            self.label_all.append(G.graph['label'] - 1) # May 23rd

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        if self.DGL:
            adj = self.adj_all[idx]
            num_nodes = adj.number_of_nodes()

            return {'seq':self.seq[idx].copy(),
                    'seq_mask':self.seq_mask_all[idx].copy(),
                    'adj':adj.clone(),
                    'feats':self.feature_all[idx].copy(),
                    'label':self.label_all[idx],
                    'num_nodes': num_nodes}
        else:
            adj = self.adj_all[idx]
            adj_dim = adj.shape[0]
            num_nodes = adj.shape[1]
            adj_padded = np.zeros((adj_dim, self.max_num_nodes, self.max_num_nodes))
            adj_padded[:, :num_nodes, :num_nodes] = adj

            return {'seq':self.seq[idx].copy(),
                    'seq_mask':self.seq_mask_all[idx].copy(),
                    'adj':adj_padded,
                    'feats':self.feature_all[idx].copy(),
                    'label':self.label_all[idx],
                    'num_nodes': num_nodes}
