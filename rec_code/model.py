import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
import torch.nn.functional as F
import utils



class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)
    
    
class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma


class CoLaKG(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset, 
                 adj_matrix=None, 
                 semantic_emb=None, 
                 user_semantic_emb=None,):
        super(CoLaKG, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.adj_matrix = adj_matrix.to(world.device)
        self.semantic_emb = semantic_emb.to(world.device)
   
        self.user_semantic_emb = user_semantic_emb.to(world.device)
        self.semantic_hid = 32
        self.dropout_i = self.config['dropout_i']
        self.dropout_u = self.config['dropout_u']
        self.dropout_neighbor = self.config['dropout_n']
        self.use_semantic_gate = bool(self.config['use_semantic_gate'])
        self.layer_cl_reg = self.config['layer_cl_reg']
        self.layer_cl_temp = self.config['layer_cl_temp']
        self.semantic_cl_reg = self.config['semantic_cl_reg']
        self.semantic_cl_temp = self.config['semantic_cl_temp']
        self.sccf_reg = self.config['sccf_reg']
        self.sccf_temp = self.config['sccf_temp']
        self.use_layer_weight = bool(self.config['use_layer_weight'])
        self.use_item_bias = bool(self.config['use_item_bias'])
        self.multi_neg_loss = bool(self.config['multi_neg_loss'])
        self.multi_neg_tau = self.config['multi_neg_tau']
        self.use_social_graph = bool(self.config['use_social_graph'])
        self.social_alpha = self.config['social_alpha']
        self.use_itemknn_score = bool(self.config['use_itemknn_score'])
        self.itemknn_alpha = self.config['itemknn_alpha']
        self.itemknn_train_alpha = self.config['itemknn_train_alpha']
        self.use_item_graph_prop = bool(self.config['use_item_graph_prop'])
        self.item_graph_alpha = self.config['item_graph_alpha']
        self.use_neighbor_cf_prior = bool(self.config['use_neighbor_cf_prior'])
        self.neighbor_cf_alpha = self.config['neighbor_cf_alpha']
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        print("self.num_items", self.num_items)
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.use_layer_weight:
            self.layer_weights = nn.Parameter(torch.zeros(self.n_layers + 1))
        if self.use_item_bias:
            self.item_bias = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=1)
            nn.init.zeros_(self.item_bias.weight)

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        world.cprint('use NORMAL distribution initilizer')
   
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        self.SocialGraph = self.dataset.getSocialGraph() if self.use_social_graph else None
        self.ItemKnnScores = self.dataset.getItemKnnScores(self.config['itemknn_k']) if self.use_itemknn_score else None
        self.ItemSemanticGraph = self._build_item_semantic_graph() if self.use_item_graph_prop else None
        self.NeighborCfPrior = self._build_neighbor_cf_prior() if self.use_neighbor_cf_prior else None
        self.semantic_map = nn.Linear(1024, self.latent_dim)
        self.user_semantic_map = nn.Linear(1024, self.latent_dim)
        if self.use_semantic_gate:
            self.item_semantic_gate = nn.Linear(self.latent_dim * 2, self.latent_dim)
            self.user_semantic_gate = nn.Linear(self.latent_dim * 2, self.latent_dim)
            self.neighbor_gate = nn.Linear(self.latent_dim * 2, self.latent_dim)
            nn.init.xavier_uniform_(self.item_semantic_gate.weight)
            nn.init.xavier_uniform_(self.user_semantic_gate.weight)
            nn.init.xavier_uniform_(self.neighbor_gate.weight)
            nn.init.constant_(self.item_semantic_gate.bias, 1.0)
            nn.init.constant_(self.user_semantic_gate.bias, 1.0)
            nn.init.constant_(self.neighbor_gate.bias, 0.0)
        print(f"lgn is already to go(drop_edge:{self.config['use_drop_edge']})")
        self.W = nn.Parameter(torch.empty(size=(1024, 32)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*32, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.W_u = nn.Parameter(torch.empty(size=(1024, 32)))
        nn.init.xavier_uniform_(self.W_u.data, gain=1.414)
        self.a_u = nn.Parameter(torch.empty(size=(2*32, 1)))
        nn.init.xavier_uniform_(self.a_u.data, gain=1.414)
        self.alpha=0.2
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def _adaptive_fusion(self, base_emb, aux_emb, gate_layer):
        gate = torch.sigmoid(gate_layer(torch.cat([base_emb, aux_emb], dim=-1)))
        return gate * base_emb + (1.0 - gate) * aux_emb

    def _build_item_semantic_graph(self):
        with torch.no_grad():
            row_index = torch.arange(self.num_items, device=world.device).unsqueeze(1).repeat(1, self.adj_matrix.shape[1])
            col_index = self.adj_matrix
            semantic_norm = F.normalize(self.semantic_emb, dim=-1)
            center = semantic_norm[row_index.reshape(-1)]
            neighbor = semantic_norm[col_index.reshape(-1)]
            weights = torch.sum(center * neighbor, dim=-1).view(self.num_items, -1)
            weights = F.softmax(weights, dim=1).reshape(-1)
            indices = torch.stack([row_index.reshape(-1), col_index.reshape(-1)])
            graph = torch.sparse.FloatTensor(indices, weights, torch.Size([self.num_items, self.num_items]))
            return graph.coalesce().to(world.device)

    def _build_neighbor_cf_prior(self):
        train_matrix = getattr(self.dataset, "UserItemNet", None)
        if train_matrix is None:
            return None

        adj = self.adj_matrix.detach().cpu().numpy()
        rows = np.arange(self.num_items, dtype=np.int64).reshape(-1, 1)
        cols = adj.astype(np.int64)

        item_degree = np.array(train_matrix.sum(axis=0)).squeeze().astype(np.float32)
        item_degree[item_degree == 0.] = 1.
        co_mat = (train_matrix.T @ train_matrix).astype(np.float32).tocsr()
        pair_rows = np.repeat(np.arange(self.num_items, dtype=np.int64), cols.shape[1])
        pair_cols = cols.reshape(-1)
        co_values = np.asarray(co_mat[pair_rows, pair_cols]).reshape(-1)
        co_values = co_values.reshape(self.num_items, cols.shape[1])
        denom = np.sqrt(item_degree[rows] * item_degree[cols])
        cf_sim = co_values / np.maximum(denom, 1e-8)
        row_sum = cf_sim.sum(axis=1, keepdims=True)
        uniform = np.full_like(cf_sim, 1.0 / cf_sim.shape[1], dtype=np.float32)
        prior = np.divide(cf_sim, row_sum, out=uniform, where=row_sum > 0)
        nonzero_ratio = float(np.mean(cf_sim > 0))
        print(f"built train-only collaborative prior for semantic neighbors, nonzero_ratio={nonzero_ratio:.4f}")
        return torch.FloatTensor(prior).to(world.device)

    def _batch_info_nce(self, anchor, positive, temperature):
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        logits = torch.matmul(anchor, positive.t()) / temperature
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0].detach()
        labels = torch.arange(anchor.shape[0], device=anchor.device)
        return F.cross_entropy(logits, labels)

    def _layer_contrastive_loss(self, layer_embs, users, pos_items):
        if len(layer_embs) < 2:
            return torch.zeros([], device=users.device)

        center_emb = layer_embs[0]
        propagated_emb = torch.mean(torch.stack(layer_embs[1:], dim=1), dim=1)
        center_users, center_items = torch.split(center_emb, [self.num_users, self.num_items])
        propagated_users, propagated_items = torch.split(propagated_emb, [self.num_users, self.num_items])

        batch_users = torch.unique(users)
        batch_items = torch.unique(pos_items)
        user_loss = self._batch_info_nce(
            propagated_users[batch_users],
            center_users[batch_users],
            self.layer_cl_temp,
        )
        item_loss = self._batch_info_nce(
            propagated_items[batch_items],
            center_items[batch_items],
            self.layer_cl_temp,
        )
        return 0.5 * (user_loss + item_loss)

    def _semantic_contrastive_loss(self, all_users, all_items, users, pos_items):
        batch_users = torch.unique(users)
        batch_items = torch.unique(pos_items)
        user_semantic = F.elu(self.user_semantic_map(self.user_semantic_emb))[batch_users]
        item_semantic = F.elu(self.semantic_map(self.semantic_emb))[batch_items]

        user_loss = self._batch_info_nce(
            all_users[batch_users],
            user_semantic,
            self.semantic_cl_temp,
        )
        item_loss = self._batch_info_nce(
            all_items[batch_items],
            item_semantic,
            self.semantic_cl_temp,
        )
        return 0.5 * (user_loss + item_loss)

    def _sccf_contrastive_loss(self, users_emb, pos_emb):
        users_emb = F.normalize(users_emb, dim=-1)
        pos_emb = F.normalize(pos_emb, dim=-1)
        pos_score = torch.sum(users_emb * pos_emb, dim=1)
        up_score = torch.exp(pos_score / self.sccf_temp) + torch.exp((pos_score ** 2) / self.sccf_temp)
        up = torch.log(up_score + 1e-8).mean()
        sim_mat = torch.matmul(users_emb, pos_emb.t())
        down_score = torch.exp(sim_mat / self.sccf_temp) + torch.exp((sim_mat ** 2) / self.sccf_temp)
        down = torch.log(down_score.mean() + 1e-8)
        return -up + down
    
    def computer(self, return_layers=False):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        
        items_semantic_emb = F.dropout(self.semantic_emb, self.dropout_i, training=self.training)
        items_semantic_emb = self.semantic_map(items_semantic_emb)
        items_semantic_emb = F.elu(items_semantic_emb)
        items_semantic_emb = F.dropout(items_semantic_emb, self.dropout_i, training=self.training)
        if self.use_semantic_gate:
            items_emb_merged = self._adaptive_fusion(items_emb, items_semantic_emb, self.item_semantic_gate)
        else:
            items_emb_merged = (items_emb + items_semantic_emb) / 2
        
        user_semantic_emb = F.dropout(self.user_semantic_emb, self.dropout_u, training=self.training)
        user_semantic_emb = self.user_semantic_map(user_semantic_emb)
        user_semantic_emb = F.elu(user_semantic_emb)
        user_semantic_emb = F.dropout(user_semantic_emb, self.dropout_u, training=self.training)
        if self.use_semantic_gate:
            users_emb_merged = self._adaptive_fusion(users_emb, user_semantic_emb, self.user_semantic_gate)
        else:
            users_emb_merged = (users_emb + user_semantic_emb) / 2

        if self.SocialGraph is not None:
            social_users_emb = torch.sparse.mm(self.SocialGraph, users_emb_merged)
            users_emb_merged = (1.0 - self.social_alpha) * users_emb_merged + self.social_alpha * social_users_emb
        
        
        neighbor_emb = items_emb_merged[self.adj_matrix]
        items_semantic_emb0 = self.semantic_emb
        neighbor_semantic_emb = self.semantic_emb[self.adj_matrix]  # N,L,d1

        # x = self.attentions(neighbor_semantic_emb, neighbor_emb, items_semantic_emb0)
        h, value_emb, semantic_emb = neighbor_semantic_emb, neighbor_emb, items_semantic_emb0
        
        Wh = torch.matmul(h, self.W)  # N,L,d
        h0 = semantic_emb.unsqueeze(1).repeat(1, h.shape[1],1)  # N,L,d1
        Wh0 = torch.matmul(h0, self.W)  # N,L,d
        
        W_concat = torch.cat((Wh, Wh0), dim=-1) # N,L,2d
        
        attention = torch.matmul(W_concat, self.a).squeeze(-1) # N,L
        attention = self.leakyrelu(attention)
        attention = F.softmax(attention, dim=1) # N,L
        if self.NeighborCfPrior is not None:
            attention = (1.0 - self.neighbor_cf_alpha) * attention + self.neighbor_cf_alpha * self.NeighborCfPrior
    
        attention = F.dropout(attention, self.dropout_neighbor, training=self.training) # N,L
        attention = attention.unsqueeze(-1)
     
        h_prime = attention * value_emb

        h_prime = torch.sum(h_prime, dim=1)
        
        h_prime = F.elu(h_prime)
      

        if self.use_semantic_gate:
            items_emb_merged = self._adaptive_fusion(items_emb_merged, h_prime, self.neighbor_gate)
        else:
            items_emb_merged = (items_emb_merged + h_prime ) / 2
        
        # items_emb = F.elu(items_emb)
       
        all_emb = torch.cat([users_emb_merged, items_emb_merged])
        embs = [all_emb]
        
        if self.config['use_drop_edge']:
            if self.training:
                # print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            if self.ItemSemanticGraph is not None:
                layer_users, layer_items = torch.split(all_emb, [self.num_users, self.num_items])
                semantic_items = torch.sparse.mm(self.ItemSemanticGraph, layer_items)
                layer_items = (1.0 - self.item_graph_alpha) * layer_items + self.item_graph_alpha * semantic_items
                all_emb = torch.cat([layer_users, layer_items])
            embs.append(all_emb)
        layer_embs = embs
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        if self.use_layer_weight:
            layer_weights = F.softmax(self.layer_weights, dim=0).view(1, -1, 1)
            light_out = torch.sum(embs * layer_weights, dim=1)
        else:
            light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        if return_layers:
            return users, items, layer_embs
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = torch.matmul(users_emb, items_emb.t())
        if self.use_item_bias:
            rating = rating + self.item_bias.weight.squeeze(-1).view(1, -1)
        rating = self.f(rating)
        if self.ItemKnnScores is not None:
            rating = rating + self.itemknn_alpha * self.ItemKnnScores[users.long()]
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        
        users_emb_ego0 = self.user_semantic_map(self.user_semantic_emb)[users]
        pos_emb_ego0 = self.semantic_map(self.semantic_emb)[pos_items]
        neg_emb_ego0 = self.semantic_map(self.semantic_emb)[neg_items]
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego, pos_emb_ego0, neg_emb_ego0, users_emb_ego0
    
    def bpr_loss(self, users, pos, neg):
        users = users.long()
        pos = pos.long()
        neg = neg.long()
        use_aux_losses = self.layer_cl_reg > 0 or self.semantic_cl_reg > 0 or self.sccf_reg > 0
        if use_aux_losses:
            all_users, all_items, layer_embs = self.computer(return_layers=True)
        else:
            all_users, all_items = self.computer()
            layer_embs = None

        users_emb = all_users[users]
        pos_emb = all_items[pos]
        neg_emb = all_items[neg]
        userEmb0 = self.embedding_user(users)
        posEmb0 = self.embedding_item(pos)
        negEmb0 = self.embedding_item(neg)
        users_emb_ego0 = self.user_semantic_map(self.user_semantic_emb)[users]
        pos_emb_ego0 = self.semantic_map(self.semantic_emb)[pos]
        neg_emb_ego0 = self.semantic_map(self.semantic_emb)[neg]
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        if self.use_item_bias:
            pos_bias = self.item_bias(pos).squeeze(-1)
            pos_scores = pos_scores + pos_bias
        if self.ItemKnnScores is not None and self.itemknn_train_alpha > 0:
            pos_scores = pos_scores + self.itemknn_train_alpha * self.ItemKnnScores[users, pos]

        if neg.dim() == 2:
            neg_scores_all = torch.sum(users_emb.unsqueeze(1) * neg_emb, dim=-1)
            if self.use_item_bias:
                neg_bias_all = self.item_bias(neg).squeeze(-1)
                neg_scores_all = neg_scores_all + neg_bias_all
            if self.ItemKnnScores is not None and self.itemknn_train_alpha > 0:
                neg_scores_all = neg_scores_all + self.itemknn_train_alpha * self.ItemKnnScores[users.unsqueeze(1), neg]
            if self.multi_neg_loss:
                tau = max(self.multi_neg_tau, 1e-6)
                neg_scores = tau * torch.logsumexp(neg_scores_all / tau, dim=1)
                negEmb0 = torch.mean(negEmb0, dim=1)
                neg_emb_ego0 = torch.mean(neg_emb_ego0, dim=1)
                if self.use_item_bias:
                    neg_bias = torch.mean(neg_bias_all, dim=1)
            else:
                hard_idx = torch.argmax(neg_scores_all, dim=1)
                batch_idx = torch.arange(neg.shape[0], device=neg.device)
                neg_emb = neg_emb[batch_idx, hard_idx]
                negEmb0 = negEmb0[batch_idx, hard_idx]
                neg_emb_ego0 = neg_emb_ego0[batch_idx, hard_idx]
                neg_scores = neg_scores_all[batch_idx, hard_idx]
                if self.use_item_bias:
                    neg_bias = neg_bias_all[batch_idx, hard_idx]
        else:
            neg_scores = torch.mul(users_emb, neg_emb)
            neg_scores = torch.sum(neg_scores, dim=1)
            if self.use_item_bias:
                neg_bias = self.item_bias(neg).squeeze(-1)
                neg_scores = neg_scores + neg_bias
            if self.ItemKnnScores is not None and self.itemknn_train_alpha > 0:
                neg_scores = neg_scores + self.itemknn_train_alpha * self.ItemKnnScores[users, neg]
        reg_terms = (
            userEmb0.norm(2).pow(2) +
            posEmb0.norm(2).pow(2) +
            negEmb0.norm(2).pow(2) +
            pos_emb_ego0.norm(2).pow(2) +
            neg_emb_ego0.norm(2).pow(2) +
            users_emb_ego0.norm(2).pow(2)
        )
        if self.use_item_bias:
            reg_terms = reg_terms + pos_bias.norm(2).pow(2) + neg_bias.norm(2).pow(2)
        reg_loss = (1/2) * reg_terms / float(len(users))
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        if self.layer_cl_reg > 0:
            loss = loss + self.layer_cl_reg * self._layer_contrastive_loss(layer_embs, users, pos)
        if self.semantic_cl_reg > 0:
            loss = loss + self.semantic_cl_reg * self._semantic_contrastive_loss(all_users, all_items, users, pos)
        if self.sccf_reg > 0:
            loss = loss + self.sccf_reg * self._sccf_contrastive_loss(users_emb, pos_emb)
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        if self.use_item_bias:
            gamma = gamma + self.item_bias(items).squeeze(-1)
        if self.ItemKnnScores is not None:
            gamma = gamma + self.itemknn_alpha * self.ItemKnnScores[users, items]
        return gamma
