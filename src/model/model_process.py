from ..utils import *
from ..data_load.data_loader import *
from torch.utils.data import DataLoader


class TrainBatchProcessor():
    def __init__(self, args, kg) -> None:
        self.args = args
        self.kg = kg
        self.dataset = TrainDatasetMarginLoss(args, kg)
        self.shuffle_mode = True
        self.data_loader = DataLoader(self.dataset,
                                      shuffle=self.shuffle_mode,
                                      batch_size=int(self.args.batch_size),
                                      collate_fn=self.dataset.collate_fn,
                                      generator=torch.Generator().manual_seed(int(args.random_seed)), # use seed generator
                                      pin_memory=True
                                      ) # real memory is enough, set pin_memory=True, faster!

    def process_epoch(self, model, optimizer):
        model.train()
        if self.args.model_name == "LoraKGE_Layers" and (self.args.using_various_ranks or self.args.using_various_ranks_reverse):
            if model.lora_ent_embeddings_list != None:
                for lora_model in model.lora_ent_embeddings_list:
                    lora_model.train(True)
        """ Start training """
        total_loss = 0.0
        if self.args.record:
            loss_save_path = "/data/my_cl_kge/save/" + str(self.args.snapshot) + ".txt"
            with open(loss_save_path, "a", encoding="utf-8") as wf:
                wf.write(str(self.args.epoch))
                wf.write("\t")
        for b_id, batch in enumerate(self.data_loader):
            """ Get loss """
            bh, br, bt, by = batch
            optimizer.zero_grad()
            batch_loss = model.loss(bh.to(self.args.device),
                                    br.to(self.args.device),
                                    bt.to(self.args.device),
                                    by.to(self.args.device) if by is not None else by,
                                    ).float()
            """ updata """
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
            """ post processing """
            model.epoch_post_processing(bh.size(0))
            if self.args.record:
                with open(loss_save_path, "a", encoding="utf-8") as wf:
                    wf.write(str(batch_loss.item()))
                    wf.write("\t")
        if self.args.record:
            with open(loss_save_path, "a", encoding="utf-8") as wf:
                wf.write("\n")
        return total_loss

class DevBatchProcessor():
    def __init__(self, args, kg) -> None:
        self.args = args
        self.kg = kg
        self.batch_size = 100
        """ prepare data """
        self.dataset = TestDataset(args, kg)
        self.data_loader = DataLoader(self.dataset,
                                      shuffle=False,
                                      batch_size=self.batch_size,
                                      collate_fn=self.dataset.collate_fn,
                                      generator=torch.Generator().manual_seed(int(args.random_seed)),
                                      pin_memory=True)

    def process_epoch(self, model):
        model.eval()
        if self.args.model_name == "LoraKGE_Layers" and self.args.using_various_ranks or self.args.using_various_ranks_reverse:
            if model.lora_ent_embeddings_list != None:
                for lora_model in model.lora_ent_embeddings_list:
                    lora_model.train(False)
        num = 0
        results = {}
        hr2t = self.kg.snapshots[self.args.snapshot].hr2t_all
        """ Start evaluation """
        for batch in self.data_loader:
            # head: (batch_size, 1)
            head, relation, tail, label = batch
            head = head.to(self.args.device)
            relation = relation.to(self.args.device)
            tail = tail.to(self.args.device)
            label = label.to(self.args.device) # (batch_size, ent_num)
            num += len(head)
            stage = "Valid" if self.args.valid else "Test"
            """ Get prediction scores """
            pred = model.predict(head, relation, stage=stage) # (batch_size, num_ent)
            """ filter: """
            batch_size_range = torch.arange(pred.size()[0], device=self.args.device)
            target_pred = pred[batch_size_range, tail]
            pred = torch.where(label.bool(), -torch.ones_like(pred) * 10000000, pred)
            pred[batch_size_range, tail] = target_pred
            if self.args.predict_result and stage == "Test":
                logits_sorted, indices_sorted = torch.sort(pred, dim=-1, descending=True)
                predict_result_path = "/data2/jun/lora_clkge/save/predict_result/" + "lora_kge/" + str(self.args.snapshot) + "_" + str(self.args.snapshot_test) + ".txt"
                with open(predict_result_path, "a", encoding="utf-8") as af:
                    batch_num = len(head)
                    for i in range(batch_num):
                        top1 = indices_sorted[i][0]
                        top2 = indices_sorted[i][1]
                        top3 = indices_sorted[i][2]
                        af.write(self.kg.id2entity[head[i].detach().cpu().item()])
                        af.write("\t")
                        af.write(self.kg.id2relation[relation[i].detach().cpu().item()])
                        af.write("\t")
                        af.write(self.kg.id2entity[tail[i].detach().cpu().item()])
                        af.write("\n")
                        af.write(self.kg.id2entity[top1.detach().cpu().item()])
                        af.write("\t")
                        af.write(self.kg.id2entity[top2.detach().cpu().item()])
                        af.write("\t")
                        af.write(self.kg.id2entity[top3.detach().cpu().item()])
                        af.write("\n")
                        af.write("----------------------------------------------------------")
                        af.write("\n")
            """ rank all candidate entities """
            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[batch_size_range, tail]
            ranks = ranks.float()
            results['count'] = torch.numel(ranks) + results.get('count', 0.0)
            results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
            results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
            for k in range(10):
                results[f'hits{k + 1}'] = torch.numel(
                    ranks[ranks <= (k + 1)]
                ) + results.get(f'hits{k + 1}', 0.0)
        count = float(results['count'])
        for key, val in results.items():
            results[key] = round(val / count, 4)
        return results