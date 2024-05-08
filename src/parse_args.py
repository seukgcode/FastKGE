import argparse
parser = argparse.ArgumentParser(description="Parser For Arguments",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# paths
parser.add_argument("-save_path", dest="save_path", default="./checkpoint/", help="Path of saved models")
parser.add_argument("-log_path", dest="log_path", default="./logs/", help="Path of saved logs")
parser.add_argument("-data_path", dest="data_path", default="./data/", help="Path of dataset")

# global setting
parser.add_argument("-random_seed", dest="random_seed", default=3407, help="Set random seeds")
parser.add_argument("-dataset", dest="dataset", default="ENTITY", help="dataset name")
parser.add_argument("-gpu", dest="gpu", default=0, help="number of gpu")

# model setting
parser.add_argument("-model_name", dest="model_name", default="LoraKGE", help="name of model")
parser.add_argument("-batch_size", dest="batch_size", default=1024, help="Set the batch size")
parser.add_argument("-learning_rate", dest="learning_rate", default=1e-4, help="Set the learning rate")
parser.add_argument("-epoch_num", dest="epoch_num", default=200, help="Set the epoch")
parser.add_argument("-note", dest='note', default='', help='The note of log file name')
parser.add_argument("-snapshot_num", dest="snapshot_num", default=5, help="The number of snapshots")
parser.add_argument("-emb_dim", dest="emb_dim", default=200, help="embedding dimension")
parser.add_argument("-margin", dest="margin", default=8.0, help="The margin of MarginLoss")
parser.add_argument("-neg_ratio", dest="neg_ratio", default=10, help="the ratio of negtive/postive facts")
parser.add_argument("-l2", dest='l2', default=0.0, help="optimizer l2")
parser.add_argument("-num_layer", dest="num_layer", default=1, help='MAE layer')
parser.add_argument("-skip_previous", dest="skip_previous", default="False", help="Allow re-training and snapshot_only models skip previous training")
parser.add_argument("-train_new", dest="train_new", default=True, help="True: Training on new facts; False: Training on all seen facts")
parser.add_argument("-valid_metrics", dest="valid_metrics", default="mrr")
parser.add_argument("-patience", dest="patience", default=3, help="early stop step")

# new updates
parser.add_argument("-debug", dest="debug", default=False, help="test mode")
parser.add_argument("-record", dest="record", default=False, help="Record the loss of different layers")
parser.add_argument("-predict_result", dest="predict_result", default=False, help="The result of predict")
parser.add_argument("-r", dest="r", default=100, help="The rank of lora")
parser.add_argument("-ent_r", dest="ent_r", default=100, help="The rank of ent lora")
parser.add_argument("-rel_r", dest="rel_r", default=10, help="The rank of rel lora")
parser.add_argument("-r_fixed", dest="r_fixed", default=True, help="fix the r")
parser.add_argument("-using_multi_layers", dest="using_multi_layers", default=False, help="Use multi_layers or not")
parser.add_argument("-multi_layers_path", dest="multi_layers_path", default="train_sorted_by_edges_betweenness.txt", help="New_path")
parser.add_argument("-num_ent_layers", dest="num_ent_layers", default=10, help="The length of ent embeddings list")
parser.add_argument("-num_rel_layers", dest="num_rel_layers", default=1, help="The length of rel embeddings list")
parser.add_argument("-using_various_ranks", dest="using_various_ranks", default=False, help="Using various ranks or not")
parser.add_argument("-using_various_ranks_reverse", dest="using_various_ranks_reverse", default=False, help="Using reverse various ranks or not")
parser.add_argument("-explore", dest="explore", default=False, help="Explorable experiments")


args = parser.parse_args()