from src.utils import *

def load_fact(path):
    """ load facts from xxx.txt """
    facts = []
    with open(path, "r") as f:
        for line in f:
            line = line.split()
            h, r, t = line[0], line[1], line[2]
            facts.append((h, r, t))
    return facts

def build_edge_index(h, t):
    """ build edge_index using h and t"""
    index = [h + t, t + h]
    return torch.LongTensor(index)

class KnowledgeGraph():
    def __init__(self, args) -> None:
        self.args = args
        self.num_ent, self.num_rel = 0, 0
        self.entity2id, self.id2entity, self.relation2id, self.id2relation = {}, {}, {}, {}
        self.relationid2invid = {}
        self.snapshots = {i: Snapshot(self.args) for i in range(int(self.args.snapshot_num))}
        self.load_data()

    def load_data(self):
        """ Load data from all snapshots """
        hr2t_all = {}
        train_all, valid_all, test_all = [], [], []
        for ss_id in range(int(self.args.snapshot_num)):
            self.new_entities = set() # all entities in this snapshot
            """ Step 1: (h, r, t) """
            train_facts = load_fact(self.args.data_path + str(ss_id) + "/" + "train.txt")
            valid_facts = load_fact(self.args.data_path + str(ss_id) + "/" + "valid.txt")
            test_facts = load_fact(self.args.data_path + str(ss_id) + "/" + "test.txt")

            """ Step 2: h -> h_id, r -> r_id, t -> t_id """
            self.expend_entity_relation(train_facts)
            self.expend_entity_relation(valid_facts)
            self.expend_entity_relation(test_facts)

            """ Step 3: (h, r, t) -> (h_id, r_id, t_id) """
            train = self.fact2id(train_facts)
            valid = self.fact2id(valid_facts, order=True)
            test = self.fact2id(test_facts, order=True)
            
            """ Step 4: [h1, h2, ..., hn] (train set),
                        [r1, r2, ..., rn] (train set),
                        [t1, t2, ..., tn] (train set),
                        {(h1, r1): t1, (h2, r2): t2, ..., (hn, rn): tn}
            """
            edge_h, edge_r, edge_t = [], [], []
            edge_h, edge_r, edge_t = self.expand_kg(train, 'train', edge_h, edge_r, edge_t, hr2t_all)
            edge_h, edge_r, edge_t = self.expand_kg(valid, 'valid', edge_h, edge_r, edge_t, hr2t_all)
            edge_h, edge_r, edge_t = self.expand_kg(test, 'test', edge_h, edge_r, edge_t, hr2t_all)

            """ Step 5: Get all (h_id, r_id, t_id) """
            train_all += train
            valid_all += valid
            test_all += test

            """ Step 6: Store this snapshot """
            self.store_snapshot(ss_id, train, train_all, valid, valid_all, test, test_all, edge_h, edge_r, edge_t, hr2t_all)
            self.new_entities.clear()

    def store_snapshot(self, ss_id, train, train_all, valid, valid_all, test, test_all, edge_h, edge_r, edge_t, hr2t_all):
        """ Store num_ent, num_rel """
        self.snapshots[ss_id].num_ent = deepcopy(self.num_ent)
        self.snapshots[ss_id].num_rel = deepcopy(self.num_rel)

        """ Store (h, r, t) """
        self.snapshots[ss_id].train = deepcopy(train)
        self.snapshots[ss_id].train_all = deepcopy(train_all)
        self.snapshots[ss_id].valid = deepcopy(valid)
        self.snapshots[ss_id].valid_all = deepcopy(valid_all)
        self.snapshots[ss_id].test = deepcopy(test)
        self.snapshots[ss_id].test_all = deepcopy(test_all)

        """ Store [h1, h2, ..., hn], [r1, r2, ..., rn], [t1, t2, ..., tn] """
        """ next 3 lines for MAE """
        self.snapshots[ss_id].edge_h = deepcopy(edge_h)
        self.snapshots[ss_id].edge_r = deepcopy(edge_r)
        self.snapshots[ss_id].edge_t = deepcopy(edge_t)

        """ Store some special things """
        self.snapshots[ss_id].hr2t_all = deepcopy(hr2t_all)
        """ next 3 lines for MAE """
        self.snapshots[ss_id].edge_index = build_edge_index(edge_h, edge_t).to(self.args.device)
        self.snapshots[ss_id].edge_type = torch.cat([torch.LongTensor(edge_r), torch.LongTensor(edge_r) + 1]).to(self.args.device)
        self.snapshots[ss_id].new_entities = deepcopy(list(self.new_entities))


    def expand_kg(self, facts, split, edge_h, edge_r, edge_t, hr2t_all):
        """ Get edge_index and edge_type for GCN and hr2t_all for filter golden facts """
        def add_key2val(dict, key, val):
            """ add {key: val} to dict"""
            if key not in dict.keys():
                dict[key] = set()
            dict[key].add(val)

        for (h, r, t) in facts:
            self.new_entities.add(h)
            self.new_entities.add(t)
            if split == "train":
                """ edge_index """
                edge_h.append(h)
                edge_r.append(r)
                edge_t.append(t)
            """ hr2t """
            add_key2val(hr2t_all, (h, r), t)
            add_key2val(hr2t_all, (t, self.relationid2invid[r]), h)
        return edge_h, edge_r, edge_t

    def fact2id(self, facts, order=False):
        """ (h, r, t) -> (h_id, r_id, t_id) """
        fact_id = []
        if order:
            i = 0
            while len(fact_id) < len(facts):
                for (h, r, t) in facts:
                    if self.relation2id[r] == i:
                        fact_id.append((self.entity2id[h], self.relation2id[r], self.entity2id[t]))
                i += 2
        else:
            for (h, r, t) in facts:
                fact_id.append((self.entity2id[h], self.relation2id[r], self.entity2id[t]))
        return fact_id

    def expend_entity_relation(self, facts):
        """ extract entities and relations from new facts """
        for (h, r, t) in facts:
            """ extract entities """
            if h not in self.entity2id.keys():
                self.entity2id[h] = self.num_ent
                self.id2entity[self.num_ent] = h
                self.num_ent += 1
            if t not in self.entity2id.keys():
                self.entity2id[t] = self.num_ent
                self.id2entity[self.num_ent] = t
                self.num_ent += 1

            """ extract relations """
            if r not in self.relation2id.keys():
                self.relation2id[r] = self.num_rel
                self.id2relation[self.num_rel] = r
                self.relation2id[r + "_inv"] = self.num_rel + 1
                self.id2relation[self.num_rel + 1] = r + "_inv"
                self.relationid2invid[self.num_rel] = self.num_rel + 1
                self.relationid2invid[self.num_rel + 1] = self.num_rel
                self.num_rel += 2


class Snapshot():
    def __init__(self, args) -> None:
        self.args = args
        self.num_ent, self.num_rel = 0, 0
        self.train, self.train_all, self.valid, self.valid_all, self.test, self.test_all = [], [], [], [], [], []
        self.edge_h, self.edge_r, self.edge_t = [], [], []
        self.hr2t_all = {}
        self.edge_index, self.edge_type = None, None
        self.new_entities = []