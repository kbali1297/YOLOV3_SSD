from __future__ import division

import datetime
import sys
import time

import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader

from evaluate import evaluate
from utils.models import *
from utils.augmentations import *
from utils.datasets import *
from utils.parse_config import *
from utils.utils import *
from utils.prune import *

import argparse

parser = argparse.ArgumentParser(description='CmdLine Parser', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(  '--model', default='SSD', type=str)
parser.add_argument(  '--prune_amount', default=0.0, type=float)
parser.add_argument(  '--num_prune', default=0, type=int)
parser.add_argument(  '--epochs', default=20, type=int)
parser.add_argument(  '--batch_size', default=4, type=int)
parser.add_argument(  '--gradient_accumulations', default=2, type=int)
parser.add_argument(  '--data_config', default='config/kitti_1cls.data', type=str)
parser.add_argument(  '--n_cpu', default=1, type=int)
parser.add_argument(  '--img_size', default=300, type=int)
parser.add_argument(  '--evaluation_interval', default=1, type=int)
parser.add_argument(  '--compute_map', default=False, type=bool)
parser.add_argument(  '--multiscale_training', default=False, type=bool)
parser.add_argument(  '--verbose', default=False, type=bool)
parser.add_argument(  '--use_gpu', default=True, type=bool)
parser.add_argument(  '--load_ssd_model', default='ssd.pth', type=str)
parser.add_argument(  '--save_ssd_model', default=None, type=str)

pargs = parser.parse_args()
params = {}
params.update(vars(pargs))


if __name__ == "__main__":

    if params['model'] == 'SSD': 
        model_def = "config/ssd-kitti.cfg"
    elif params["model"] == 'YOLO':
        model_def = "config/yolov3-kitti-tiny_1cls.cfg"

    do_prune = False
    if params['num_prune'] > 0: do_prune = True
    

    for key in params:
        print(f"{key}: {params[key]}")

    # Check if GPU is available
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cuda:0' if params["use_gpu"] else 'cpu'

    # Get data configuration
    data_config = parse_data_config(params["data_config"])
    train_path = data_config["train"]
    valid_path = data_config["test"]
    try:
        class_names = load_classes(data_config["names"])
    except Exception as e:
        print(e)
        print("Download kitti dataset from https://syncandshare.lrz.de/getlink/fiAeNAvDQEaDis3Zv1AG8K/homework-4-kitti.zip "
              "and keep its structure")
        sys.exit()
    num_classes = int(data_config["classes"])

    # Quit training if dataset folder is not in correct place
    try:
        assert len(os.listdir("kitti/images/train/")) == 188, (
            "Download kitti dataset from https://syncandshare.lrz.de/getlink/fiLHHZc1Atd2Voxv4qSqnvAU/kitti "
            "and keep its structure")
        assert len(os.listdir("kitti/images/test/")) == 34, (
            "Download kitti dataset from https://syncandshare.lrz.de/getlink/fiLHHZc1Atd2Voxv4qSqnvAU/kitti "
            "and keep its structure")
        assert len(os.listdir("kitti/labels/train/")) == 188, (
            "Download kitti dataset from https://syncandshare.lrz.de/getlink/fiLHHZc1Atd2Voxv4qSqnvAU/kitti "
            "and keep its structure")
        assert len(os.listdir("kitti/labels/test/")) == 34, (
            "Download kitti dataset from https://syncandshare.lrz.de/getlink/fiLHHZc1Atd2Voxv4qSqnvAU/kitti "
            "and keep its structure")
    except Exception as e:
        print(e)
        print("Download kitti dataset from https://syncandshare.lrz.de/getlink/fiLHHZc1Atd2Voxv4qSqnvAU/kitti "
              "and keep its structure")
        sys.exit()

    # Quit training if train.txt are not in dataset folder
    try:
        open(train_path)
    except Exception as e:
        print(e)
        print("No train.txt found in kitti directory. Place your created train.txt from task1 in the kitti folder!")
        sys.exit()

    # Get dataloader
    dataset = ListDataset(train_path, multiscale=params["multiscale_training"], img_size=params["img_size"],
                            transform=AUGMENTATION_TRANSFORMS, num_classes=num_classes)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=params["batch_size"],
            shuffle=True,
            num_workers=params["n_cpu"],
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

    model_type = model_def.split('/')[1].split('-')[0]
    
    if model_type == 'yolov3':
        # Initiate model
        model = Darknet(model_def, img_size=params["img_size"]).to(device)
        model.apply(weights_init_normal)
        
        # Load model
        if os.path.exists('yolov3.pth'):
            print(f'Loading saved model....')
            model.load_state_dict(torch.load('yolov3.pth'))
        
        

        # Define optimizer
        lr = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Define metrics
        metrics = [
            "grid_size",
            "loss",
            "x",
            "y",
            "w",
            "h",
            "conf",
            "cls",
            "cls_acc",
            "recall50",
            "recall75",
            "precision",
            "conf_obj",
            "conf_noobj",
        ]

        # Training loop
        for epoch in range(params["epochs"]):
            model.train()
            start_time = time.time()
            train_loss = 0
            # select smaller batches and train batch-by-batch
            for batch_i, (img_path, imgs, targets) in enumerate(tqdm.tqdm(dataloader,
                                                                        desc=f"Training Epoch {epoch}/{params['epochs']}")):
                batches_done = len(dataloader) * epoch + batch_i

                imgs = Variable(imgs.to(device))
                targets = Variable(targets.to(device), requires_grad=False)

                # Inference and backpropagation
                loss, outputs = model(imgs, targets)
                loss.backward()

                train_loss += loss.item()

                if batches_done % params["gradient_accumulations"] == 0:
                    # Accumulates gradient before each step
                    optimizer.step()
                    optimizer.zero_grad()

                # ----------------
                #   Log progress
                # ----------------

                log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, params["epochs"], batch_i, len(dataloader))

                metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

                # Log metrics at each YOLO layer
                for i, metric in enumerate(metrics):
                    formats = {m: "%.6f" for m in metrics}
                    formats["grid_size"] = "%2d"
                    formats["cls_acc"] = "%.2f%%"
                    row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                    metric_table += [[metric, *row_metrics]]

                log_str += f"\nTotal loss {to_cpu(loss).item()}"

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"train/{name}_{j + 1}", metric)]
                tensorboard_log += [("train/loss", to_cpu(loss).item())]

                # Determine approximate time left for epoch
                epoch_batches_left = len(dataloader) - (batch_i + 1)
                time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
                log_str += f"\n---- ETA {time_left}"

                if params["verbose"]:
                    print(log_str)

                model.seen += imgs.size(0)

            print(f'Epoch {epoch} Loss: {train_loss}')
            # Evaluate model after finishing an epoch
            if epoch % params["evaluation_interval"] == 0:
                print("---- Evaluating Model ----")
                # Evaluate the model on the validation set
                metrics_output = evaluate(
                    model,
                    path=valid_path,
                    iou_thres=0.5,
                    conf_thres=0.5,
                    nms_thres=0.5,
                    img_size=params["img_size"],
                    batch_size=4,
                    num_classes=num_classes,
                    model_type='yolov3'
                )

                if metrics_output is not None:
                    precision, recall, AP, f1, ap_class = metrics_output
                    evaluation_metrics = [
                        ("validation/precision", precision.mean()),
                        ("validation/recall", recall.mean()),
                        ("validation/mAP", AP.mean()),
                        ("validation/f1", f1.mean()),
                    ]
                    print(f'Epoch: {epoch}, Precision: {precision.mean():.2f},  Recall: {recall.mean():.2f}, mAP: {AP.mean():.2f}, F1: {f1.mean():.2f}')

                    print(f"---- AP class Car: {round(AP.mean(), 2)}")
                else:
                    print("---- AP not measured (no detections found by model)")

                torch.save(model.state_dict(), "yolov3.pth")
                print(f"Epoch {epoch} finished! Saving model at yolov3.pth\n\n\n")

    elif model_type=="ssd":
        # Instantiate Model
        model = SSD(model_def, num_classes=num_classes).to(device)
        load_exception = False ## Exceptions encountered while loading model
        # Load Model
        if os.path.exists(params['load_ssd_model']):
            print(f'Loading saved model {params["load_ssd_model"]}....')
            
            model.load_state_dict(torch.load(params['load_ssd_model']))
            #except: ## Pruned model instance
            #    print("could not load model")
            #    exit(1)
                #load_exception = True
                #model, _ = prune_model(model, amount=0, dim=0, norm=2)
                #model.load_state_dict(torch.load('ssd_pruned_pasted.pth'))
                
        if params['save_ssd_model'] == None: params['save_ssd_model'] = params['load_ssd_model']

        loss_module = MultiBoxLoss(num_classes=num_classes, overlap_thresh=0.5, 
                                   prior_for_matching=True, bkg_label=0, neg_mining=True, 
                                   neg_pos=3, neg_overlap=0.5, encode_target=False, variances=model.variances)
        
        lr=1e-4
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        num_pruned = 0
        for epoch in range(params["epochs"]):

            model.train()
            start_time = time.time()
            continue_flag=0
            train_loss = 0

            if do_prune and num_pruned<params['num_prune']:
                model, percent_pruned = prune_model(model, params['prune_amount'], 0, 2)
                num_pruned+=1
                print(f'Non Pruned Params: {percent_pruned * 100}%')
            # if (epoch+1)%40 == 0:
            #     lr *= 0.9
            #     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            for batch_i, (img_path, img, targets) in enumerate(tqdm.tqdm(dataloader,
                                                            desc=f"Training Epoch {epoch}/{params['epochs']}")):

                #print(f'Input Img batch shape: {img.shape}')
                #print(f'Target Img batch shape: {targets.shape}')
                predictions = model(img.to(device), 'train')
                targets = Variable(targets).to(device)

                targets_ssd = [None] * img.shape[0]
                for k in range(targets.shape[0]):
                    if targets_ssd[int(targets[k,0])] == None:
                        targets_ssd[int(targets[k,0])] = targets[k:k+1,1:]
                    else:
                        targets_ssd[int(targets[k,0])] = \
                                torch.cat([targets_ssd[int(targets[k,0])], \
                                            targets[k:k+1,1:]], dim=0)

                for i in range(len(targets_ssd)):
                    if targets_ssd[i] == None:
                        print(f' check {i} : {img_path[i]}')
                        continue_flag=1

                if continue_flag==1: 
                    continue_flag=0
                    continue

                loss = loss_module(predictions, targets_ssd)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                # ----------------
                #   Log progress
                # ----------------

                log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, params["epochs"], batch_i, len(dataloader))

                log_str += f"\nTotal loss {to_cpu(loss).item()}"

                # Determine approximate time left for epoch
                epoch_batches_left = len(dataloader) - (batch_i + 1)
                time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
                log_str += f"\n---- ETA {time_left}"

                if params["verbose"]:
                    print(log_str)

            print(f'Epoch {epoch} Loss: {train_loss:.2f}')
            # Evaluate model after finishing an epoch
            if epoch % params["evaluation_interval"] == 0:
                print("---- Evaluating Model ----")
                # Evaluate the model on the validation set
                metrics_output = evaluate(
                    model,
                    path=valid_path,
                    iou_thres=0.5,
                    conf_thres=0.5,
                    nms_thres=0.5,
                    img_size=params["img_size"],
                    batch_size=4,
                    num_classes=num_classes,
                    model_type='ssd'
                )

                if metrics_output is not None:
                    precision, recall, AP, f1, ap_class = metrics_output
                    evaluation_metrics = [
                        ("validation/precision", precision.mean()),
                        ("validation/recall", recall.mean()),
                        ("validation/mAP", AP.mean()),
                        ("validation/f1", f1.mean()),
                    ]

                    print(f'Epoch: {epoch}, Precision: {precision.mean():.2f},  Recall: {recall.mean():.2f}, mAP: {AP.mean():.2f}, F1: {f1.mean():.2f}')

                    print(f"---- AP class Car: {round(AP.mean(), 2)}")
                else:
                    print("---- AP not measured (no detections found by model)")

                torch.save(model.state_dict(), params['save_ssd_model'])
                print(f"Epoch {epoch} finished! Saving model at {params['save_ssd_model']}\n\n\n") 

        ## Save Pruned model if do_prune
        if do_prune or load_exception:
            model = remove_pruning(model)
            torch.save(model.state_dict(), f'{params["save_ssd_model"][:-4]}_perm_pruned.pth')    








