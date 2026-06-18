import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
import datetime
import os
from os.path import join
import register
from register import dataset
from sklearn.metrics.pairwise import cosine_similarity

utils.set_seed(world.seed)
print(">>SEED:", world.seed)
experiment_start_time = time.time()

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
k = world.config['neighbor_k']

log_file = f"../logs/{world.dataset}_{world.model_name}_neighbor{str(k)}_{current_time}.txt"
os.makedirs(os.path.dirname(log_file), exist_ok=True)

item_semantic_emb = torch.load(world.item_semantic_emb_file)
user_semantic_emb = torch.load(world.user_semantic_emb_file)
cosine_sim_matrix = cosine_similarity(item_semantic_emb.numpy())
sorted_indices = np.argsort(-cosine_sim_matrix, axis=1)
sorted_indices = sorted_indices[:, 1:k+1] # does not include itself
sorted_indices = torch.tensor(sorted_indices).long()


Recmodel = register.MODELS[world.model_name](world.config, dataset, sorted_indices, item_semantic_emb, user_semantic_emb)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")
    

with open(log_file, "w") as f:
    f.write("Training Log\n")
    f.write("====================\n")

best_test_records = {
    metric: [
        {"epoch": None, "value": -np.inf, "result": None}
        for _ in world.topks
    ]
    for metric in ["precision", "recall", "ndcg"]
}


def _format_array(values):
    return "[" + ", ".join(f"{float(v):.8f}" for v in values) + "]"


def update_best_test_records(results, epoch_num):
    for metric, records in best_test_records.items():
        for i, record in enumerate(records):
            value = float(results[metric][i])
            if value > record["value"]:
                record["epoch"] = epoch_num
                record["value"] = value
                record["result"] = {
                    "precision": results["precision"].copy(),
                    "recall": results["recall"].copy(),
                    "ndcg": results["ndcg"].copy(),
                }


def format_best_test_records():
    has_result = any(
        record["epoch"] is not None
        for records in best_test_records.values()
        for record in records
    )
    if not has_result:
        return ["BEST TEST RESULTS: No test results were recorded."]

    lines = ["BEST TEST RESULTS"]
    primary_index = len(world.topks) - 1
    primary_k = world.topks[primary_index]
    primary_record = best_test_records["ndcg"][primary_index]
    primary_result = primary_record["result"]
    lines.append(
        f"PRIMARY BEST NDCG@{primary_k}: EPOCH[{primary_record['epoch']}/{world.TRAIN_epochs}] "
        f"value={primary_record['value']:.8f} "
        f"precision={_format_array(primary_result['precision'])} "
        f"recall={_format_array(primary_result['recall'])} "
        f"ndcg={_format_array(primary_result['ndcg'])}"
    )

    for metric, records in best_test_records.items():
        for i, record in enumerate(records):
            result = record["result"]
            lines.append(
                f"BEST {metric.upper()}@{world.topks[i]}: EPOCH[{record['epoch']}/{world.TRAIN_epochs}] "
                f"value={record['value']:.8f} "
                f"precision={_format_array(result['precision'])} "
                f"recall={_format_array(result['recall'])} "
                f"ndcg={_format_array(result['ndcg'])}"
            )
    return lines

try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        
        if epoch % 5 == 0:
            cprint("[TEST]")
            test_results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            update_best_test_records(test_results, epoch + 1)
            log_message = f'TEST RESULTS at EPOCH[{epoch+1}/{world.TRAIN_epochs}]: {test_results}'
            print(log_message)
            with open(log_file, "a") as f:
                f.write(log_message + "\n")
        
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
        
        end = time.time()
        epoch_time = end - start
        
        log_message = f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information} - Time: {epoch_time:.2f} seconds'
        print(log_message)
        
        with open(log_file, "a") as f:
            f.write(log_message + "\n")
        
        torch.save(Recmodel.state_dict(), weight_file)

finally:
    elapsed_time = time.time() - experiment_start_time
    elapsed_delta = datetime.timedelta(seconds=int(elapsed_time))
    elapsed_message = f"TOTAL EXPERIMENT TIME: {elapsed_time:.2f} seconds ({elapsed_delta})"
    final_messages = [elapsed_message] + format_best_test_records()
    for message in final_messages:
        print(message)
    with open(log_file, "a") as f:
        for message in final_messages:
            f.write(message + "\n")
    if world.tensorboard:
        w.close()
