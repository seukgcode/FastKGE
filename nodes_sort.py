"""
3. Generate the distance and the node degree of every entity e ==> train_distance_nodes
"""

import os

def nodes_sort():
    dataset_names = ["FB_CKGE", "WN_CKGE"]
    base_path = "./data/"
    for dataset_name in dataset_names:
        seen_entities = set()
        true_path = os.path.join(base_path, dataset_name)
        for snapshot in range(5):
            entity2id = {}
            entities_sorted = {}
            true_path_snapshot = os.path.join(true_path, str(snapshot))
            entity2id_path = os.path.join(true_path_snapshot, "entity2id.txt")
            with open(entity2id_path, "r", encoding="utf-8") as rf:
                lines = list(rf.readlines())
                for line in lines:
                    line_split = line.strip().split()
                    ent, ent_id = line_split[0], line_split[1]
                    entity2id[ent] = int(ent_id)
            train_path = os.path.join(true_path_snapshot, "train.txt")
            with open(train_path, "r", encoding="utf-8") as rf:
                lines = list(rf.readlines())
                if snapshot == 0:
                    for line in lines:
                        line_split = line.strip().split("\t")
                        head, relation, tail = line_split[0], line_split[1], line_split[2]
                        seen_entities.add(entity2id[head])
                        seen_entities.add(entity2id[tail])
                else:
                    for entity in seen_entities:
                        entities_sorted[entity] = (0, 0)
                    entity_distance = 0
                    while True:
                        last_seen_entites_length = len(seen_entities)
                        cnt = 0
                        for line in lines:
                            line_split = line.strip().split("\t")
                            head, relation, tail = line_split[0], line_split[1], line_split[2]
                            if entity2id[head] in seen_entities and entity2id[tail] not in seen_entities:
                                entities_sorted[entity2id[tail]] = (entity_distance + 1, 0)
                                seen_entities.add(entity2id[tail])
                            elif entity2id[head] not in seen_entities and entity2id[tail] in seen_entities:
                                entities_sorted[entity2id[head]] = (entity_distance + 1, 0)
                                seen_entities.add(entity2id[head])
                            else:
                                cnt += 1
                        if last_seen_entites_length == len(seen_entities):
                            break
                        else:
                            entity_distance += 1
                    nodes_degree_path = os.path.join(true_path_snapshot, "train_nodes_degree.txt")
                    with open(nodes_degree_path, "r", encoding="utf-8") as rrf:
                        lines = list(rrf.readlines())
                        for line in lines:
                            line_split = line.strip().split("\t")
                            entity, nodes = line_split[0], line_split[1]
                            if int(entity) in entities_sorted:
                                entities_sorted[int(entity)] = (entities_sorted[int(entity)][0], float(nodes))
                            else:
                                entities_sorted[int(entity)] = (100, float(nodes))
            train_distance_nodes_path = os.path.join(true_path_snapshot, "train_distance_nodes.txt")
            with open(train_distance_nodes_path, "w", encoding="utf-8") as wf:
                for key, value in entities_sorted.items():
                    wf.write(str(key))
                    wf.write("\t")
                    wf.write(str(value[0]))
                    wf.write("\t")
                    wf.write(str(value[1]))
                    wf.write("\n")
    print("ok")

if __name__ == "__main__":
    nodes_sort()