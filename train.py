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
## Add argparse to toggle between models 

if __name__ == "__main__":
    opt = dict()
    opt["epochs"] = 100
    opt["batch_size"] = 4
    opt["gradient_accumulations"] = 2
    opt["model_def"] = "config/ssd-kitti.cfg"
    opt["data_config"] = "config/kitti_1cls.data"
    opt["n_cpu"] = 1
    opt["img_size"] = 300
    opt["evaluation_interval"] = 1
    opt["compute_map"] = False
    opt["multiscale_training"] = False
    opt["verbose"] = False
    opt["use_gpu"] = True

    for key in opt:
        print(f"{key}: {opt[key]}")

    # Check if GPU is available
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cuda:0' if opt["use_gpu"] else 'cpu'

    # Get data configuration
    data_config = parse_data_config(opt["data_config"])
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
    dataset = ListDataset(train_path, multiscale=opt["multiscale_training"], img_size=opt["img_size"],
                            transform=AUGMENTATION_TRANSFORMS, num_classes=num_classes)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt["batch_size"],
            shuffle=True,
            num_workers=opt["n_cpu"],
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

    model_type = opt["model_def"].split('/')[1].split('-')[0]
    
    if model_type == 'yolov3':
        # Initiate model
        model = Darknet(opt["model_def"], img_size=opt["img_size"]).to(device)
        model.apply(weights_init_normal)

        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters())

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
        for epoch in range(opt["epochs"]):
            model.train()
            start_time = time.time()
            train_loss = 0
            # select smaller batches and train batch-by-batch
            for batch_i, (img_path, imgs, targets) in enumerate(tqdm.tqdm(dataloader,
                                                                        desc=f"Training Epoch {epoch}/{opt['epochs']}")):
                batches_done = len(dataloader) * epoch + batch_i

                imgs = Variable(imgs.to(device))
                targets = Variable(targets.to(device), requires_grad=False)

                # Inference and backpropagation
                loss, outputs = model(imgs, targets)
                loss.backward()

                train_loss += loss.item()

                if batches_done % opt["gradient_accumulations"] == 0:
                    # Accumulates gradient before each step
                    optimizer.step()
                    optimizer.zero_grad()

                # ----------------
                #   Log progress
                # ----------------

                log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt["epochs"], batch_i, len(dataloader))

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

                if opt["verbose"]:
                    print(log_str)

                model.seen += imgs.size(0)

            print(f'Epoch {epoch} Loss: {train_loss}')
            # Evaluate model after finishing an epoch
            if epoch % opt["evaluation_interval"] == 0:
                print("---- Evaluating Model ----")
                # Evaluate the model on the validation set
                metrics_output = evaluate(
                    model,
                    path=valid_path,
                    iou_thres=0.5,
                    conf_thres=0.5,
                    nms_thres=0.5,
                    img_size=opt["img_size"],
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

                    print(f"---- AP class Car: {round(AP.mean(), 2)}")
                else:
                    print("---- AP not measured (no detections found by model)")

                torch.save(model.state_dict(), "yolov3.pth")
                print(f"Epoch {epoch} finished! Saving model at yolov3.pth\n\n\n")

    elif model_type=="ssd":
        model = SSD(opt["model_def"], num_classes=num_classes).to(device)
        loss_module = MultiBoxLoss(num_classes=num_classes, overlap_thresh=0.5, 
                                   prior_for_matching=True, bkg_label=0, neg_mining=True, 
                                   neg_pos=3, neg_overlap=0.5, encode_target=False, variances=model.variances)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(opt["epochs"]):
            model.train()
            start_time = time.time()
            continue_flag=0
            train_loss = 0
            for batch_i, (img_path, img, targets) in enumerate(tqdm.tqdm(dataloader,
                                                            desc=f"Training Epoch {epoch}/{opt['epochs']}")):

                #print(f'Input Img batch shape: {img.shape}')
                #print(f'Target Img batch shape: {targets.shape}')
                predictions = model(img.to(device))
                targets = Variable(targets).to(device)

                # if batch_i !=2 : 
                #     continue
                # else:
                #     print(targets)
                # print(f'targets shape: {targets.shape}')
                #print(f'predictions shape: {predictions.shape}')

                targets_ssd = [None] * img.shape[0]
                for k in range(targets.shape[0]):
                    #print(targets[k,0])
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
                #print(f'targets: {targets}')
                #print(f'targets_ssd: {targets_ssd}')

                loss = loss_module(predictions, targets_ssd)
                #try:
                #    loss_loc, loss_conf = loss_module(predictions, targets_ssd)
                #except:
                #    print('One of the targets is None')
                #    print(f'targets: {targets}')
                #    print(f'targets_ssd: {targets_ssd}')
                #    exit(1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                # ----------------
                #   Log progress
                # ----------------

                log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt["epochs"], batch_i, len(dataloader))

                log_str += f"\nTotal loss {to_cpu(loss).item()}"

                # Determine approximate time left for epoch
                epoch_batches_left = len(dataloader) - (batch_i + 1)
                time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
                log_str += f"\n---- ETA {time_left}"

                if opt["verbose"]:
                    print(log_str)

            print(f'Epoch {epoch} Loss: {train_loss:.2f}')
            # Evaluate model after finishing an epoch
            if epoch % opt["evaluation_interval"] == 0:
                print("---- Evaluating Model ----")
                # Evaluate the model on the validation set
                metrics_output = evaluate(
                    model,
                    path=valid_path,
                    iou_thres=0.5,
                    conf_thres=0.5,
                    nms_thres=0.5,
                    img_size=opt["img_size"],
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
                    print(f"---- AP class Car: {round(AP.mean(), 2)}")
                else:
                    print("---- AP not measured (no detections found by model)")

                torch.save(model.state_dict(), "ssd.pth")
                print(f"Epoch {epoch} finished! Saving model at ssd.pth\n\n\n")    








