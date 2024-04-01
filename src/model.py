import argparse
import time

import torch.nn as nn
from tqdm import tqdm
from utils import *
import torch.optim as optim
import torch.utils.data as Data
from data2 import *
from torch.optim.lr_scheduler import StepLR

logger = logging.getLogger(__name__)


class Scorpion(nn.Module):
    def __init__(self, args):
        super(Scorpion, self).__init__()
        self.n_proxy_sets = args.n_proxy_sets #h
        self.n_elements = args.n_elements #m
        self.d = args.d
        self.box_d = args.box_d
        self.args = args
        self.std_vol = -2
        self.relu = nn.ReLU()
        self.device = args.device
        self.proxy_sets = nn.Parameter(torch.FloatTensor(self.n_proxy_sets, self.n_elements, self.d))
        self.criterion = nn.CrossEntropyLoss()
        self.box_size_loss = nn.MSELoss()
        self.proxy_sets.data.uniform_(-1, 1)


    def infoNCE(self, x):
        batch_size = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([batch_size]).to(self.device).long()
        loss = self.criterion(x, label)
        return loss

    # def forward(self, x, index):  # Pooling
    #     splits = torch.split(x, index.tolist(), dim=0)
    #     pool_x = [torch.max(split, dim=0, keepdim=True)[0] for split in splits]
    #     # pool_x = [torch.mean(split, dim=0) for split in splits]
    #     pool_x = torch.stack(pool_x, dim=0).squeeze().to(self.device)
    #     out = F.normalize(pool_x, p=2, dim=1)
    #     return out

    def forward(self, x, index):  # x: (M) x d
        x = x.transpose(0, 1).to(self.device)
        sim_mat = torch.matmul(self.proxy_sets, x)  # h * m * M  m: # of elements in each P
        t, _ = torch.max(sim_mat, dim=1) #h * m * M
        # t2, _ = torch.max(sim_mat, dim=0)
        splits = torch.split(t, index.tolist(), dim=1)
        result = [torch.sum(split, dim=1) for split in splits]
        t = torch.stack(result, dim=0).to(torch.float)  # bz*h
        out = F.normalize(t, p=2, dim=1)
        return out

    def contrastive_loss(self, pos_1, pos_2, neg_value):
        bsz = pos_1.shape[0]
        l_pos = torch.bmm(pos_1.view(bsz, 1, -1), pos_2.view(bsz, -1, 1))
        l_pos = l_pos.view(bsz, 1)
        l_neg = torch.mm(pos_1.view(bsz, -1), neg_value.t())
        logits = torch.cat((l_pos, l_neg), dim=1)
        logits = logits.squeeze().contiguous()
        return self.infoNCE(logits / self.args.t)

    def contrastive_ranking_loss(self, anchor, pos_list, neg_value):
        bsz = anchor.shape[0]
        list_size = int(pos_list.shape[0]/bsz)
        l_neg = torch.mm(anchor.view(bsz, -1), neg_value.t())
        pos_list = pos_list.view(bsz, list_size, -1)
        tau = self.args.t
        loss_list = []
        for i in range(list_size):
            top_i = pos_list[:, i, :]
            neg_i = pos_list[:, i+1:, :]
            l_pos = torch.bmm(anchor.view(bsz, 1, -1), top_i.view(bsz, -1, 1))
            r_neg = torch.bmm(anchor.view(bsz, 1, -1), neg_i.permute(0, 2, 1))
            l_pos = l_pos.view(bsz, 1)
            r_neg = r_neg.squeeze().view(bsz, -1)
            logits = torch.cat((l_pos, l_neg, r_neg), dim=1)
            logits = logits.squeeze().contiguous()
            tau = tau + i/list_size * (0.1 - self.args.t)
            l_i = self.infoNCE(logits / tau)
            loss_list.append(l_i)
        return torch.mean(torch.stack(loss_list))


    def update(self, network: nn.Module):
        for key_param, query_param in zip(self.parameters(), network.parameters()):
            key_param.data *= self.args.momentum
            key_param.data += (1 - self.args.momentum) * query_param.data
        self.eval()


class Trainer(object):
    def __init__(self, seed=2024):
        # Set the random seed manually for reproducibility.
        self.seed = seed
        fix_seed(seed)
        # set
        parser = argparse.ArgumentParser()
        self.args = parse_options(parser)
        self.neg_queue = []
        self.aug_queue = []
        self.index_queue = []
        self.aug_index_queue = []
        self.aug_list_queue = []
        self.aug_list_index_queue =[]
        self.label_queue = []
        self.device = torch.device(self.args.device)
        train_dataset = MyDataset(self.args.anchor_path, self.args.auglist_path, list_size = self.args.list_size, training ='true', plm=self.args.plm)
        lake = TestDataset(self.args.test_path_mat, None, plm=self.args.plm)
        self.loader = Data.DataLoader(dataset=train_dataset,
                                     batch_size=self.args.batch_size,
                                     shuffle=True,
                                     num_workers=0,
                                     collate_fn=train_dataset.pad)

        self.loader2 = Data.DataLoader(dataset=lake,
                                     batch_size=self.args.batch_size,
                                     shuffle=False,
                                     num_workers=0,
                                     collate_fn=lake.pad)
        self.query = load_query(self.args.query)
        self.lake = np.load(self.args.test_path_mat, allow_pickle=True)
        self.model = Scorpion(self.args).to(self.device)
        self._model = Scorpion(self.args).to(self.device)
        self._model.update(self.model)
        self.lr = self.args.lr
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)
        self.scheduler = StepLR(self.optimizer, step_size=50, gamma=0.5)


    def test_without_train(self):
        with open(self.args.gt, 'r') as file:
            csv_reader = csv.reader(file)
            gt = []
            for row in csv_reader:
                gt.append(row)
        self.evaluate(-1, -1, gt)


    def train(self):
        all_anchor_batches = []
        all_pos_batches = []
        best = 0
        print("begining training.....")
        with open(self.args.gt, 'r') as file:
            csv_reader = csv.reader(file)
            gt = []
            for row in csv_reader:
                gt.append(row)
        for batch_id, (data, aug_data, aug_list, index, aug_index, aug_list_index) in enumerate(self.loader):
            all_anchor_batches.append((data, aug_data, aug_list, index, aug_index, aug_list_index))
        random.shuffle(all_anchor_batches)
        for epoch in range(self.args.epoch):
            t1 = time.time()
            # adjust_learning_rate(self.optimizer, epoch, self.lr)
            for batch_id, x in tqdm(enumerate(all_anchor_batches)):
                data, aug_data, aug_list, index, aug_index, aug_list_index  = x
                with torch.no_grad():
                    self.neg_queue.append(data)
                    self.aug_queue.append(aug_data)
                    self.aug_list_queue.append(aug_list)
                    self.index_queue.append(index)
                    self.aug_index_queue.append(aug_index)
                    self.aug_list_index_queue.append(aug_list_index)
                    # self.label_queue.append(label)
                if len(self.neg_queue) == self.args.queue_length + 1:
                    anchor_batch = self.neg_queue[0]
                    anchor_index = self.index_queue[0]
                    aug_index = self.aug_index_queue[0]
                    pos_batch = self.aug_queue[0]
                    pos_list_batch = self.aug_list_queue[0]
                    pos_list_index = self.aug_list_index_queue[0]
                    # label = self.label_queue[0]
                    self.neg_queue = self.neg_queue[1:]
                    self.index_queue = self.index_queue[1:]
                    self.aug_index_queue = self.aug_index_queue[1:]
                    self.aug_queue = self.aug_queue[1:]
                    self.aug_list_queue = self.aug_list_queue[1:]
                    self.aug_list_index_queue = self.aug_list_index_queue[1:]
                    neg_queue = torch.cat(self.neg_queue, dim=0)
                    neg_index = torch.cat(self.index_queue, dim=0)
                else:
                    continue

                self.optimizer.zero_grad()

                anchor = self.model(anchor_batch, anchor_index)
                with torch.no_grad():
                    self._model.eval()
                    pos_list = self._model(pos_list_batch, pos_list_index)
                    # pos = self._model(pos_batch, aug_index)
                    neg_value = self._model(neg_queue, neg_index)
                loss = self.model.contrastive_ranking_loss(anchor, pos_list, neg_value)
                # loss = self.model.contrastive_loss(anchor, pos, neg_value)
                loss.backward(retain_graph=True)

                self.optimizer.step()

                if batch_id == len(all_anchor_batches)-1:
                    print('epoch: {} batch: {} loss: {}'.format(epoch, batch_id, loss.item()))
                    pre = self.evaluate(epoch, batch_id, gt)
                    if pre > best:
                        # save_model(self.model, "../check/" + self.args.datasets + "/", "mixmodel_list=5")
                        # input the number of version
                        save_model(self.model, "../check/"+self.args.datasets+"/", "version_num")
                        best = pre
                self._model.update(self.model)
                del anchor
                del pos_batch
                del pos_list
                del neg_value
            # self.scheduler.step()


    def evaluate(self, epoch, batch_id, gt):
        print("Evaluate at epoch {} at batch_id {}...".format(epoch, batch_id))
        with torch.no_grad():
            self.model.eval()
            lake = None
            for batch_id, (data, index) in enumerate(self.loader2):
                emb = self.model(data, index)
                if lake == None:
                    lake = emb
                else:
                    lake = torch.cat((lake, emb), dim=0)

            query = np.concatenate(self.query, axis=0)
            index = list()
            for col in self.query:
                index.append(col.shape[0])
            query = self.model(torch.tensor(query).to(self.device), torch.tensor(index))

            topk = 25
            retrieval_mat = torch.mm(query, lake.transpose(0, 1))
            topk_values, topk_indices = torch.topk(retrieval_mat, k=topk, dim=1)

            hit = 0
            for i in range(len(gt)):
                hit += sum([1 for x in topk_indices[i] if str(x.item()) in gt[i][:25]])
            print("Hit:", hit)
            pre = round(hit * 100.0 / (len(gt) * 25), 2)
            print("Recall@{}: {}%".format(topk, pre))
            return pre


    def search(self, model_path, topk):
        with open(self.args.gt, 'r') as file:
            csv_reader = csv.reader(file)
            gt = []
            for row in csv_reader:
                gt.append(row)
        model = Scorpion(self.args).to(self.device)
        model.load_state_dict(torch.load(model_path))

        with torch.no_grad():
            model.eval()
            lake = None
            for batch_id, (data, index) in enumerate(self.loader2):
                emb = model(data, index)
                if lake == None:
                    lake = emb
                else:
                    lake = torch.cat((lake, emb), dim=0)

            query = np.concatenate(self.query, axis=0)
            index = list()
            for col in self.query:
                index.append(col.shape[0])
            query = model(torch.tensor(query).to(self.device), torch.tensor(index))
            similarity_mat = torch.mm(query, lake.transpose(0, 1))
            _, topk_indices = torch.topk(similarity_mat, k=topk, dim=1)
            print(topk_indices)
            for j in range(topk):
                kk = j + 1
                hit = 0
                for i in range(len(gt)):
                    hit += sum([1 for x in topk_indices[i][:kk] if str(x.item()) in gt[i][:kk]])
                # print("Hit:", hit)
                pre = round(hit * 100.0 / (len(gt) * kk), 2)
                print("Recall@{}: {}%".format(kk, pre))

            ndcg = cal_NDCG(self.query, self.lake, topk_indices, gt, 25)
            print("NDCG@25: {}".format(ndcg))










