import torch
import time
import numpy as np
from .constants import *
from utilities.device import get_device
from .lr_scheduling import get_lr
import tqdm
from dataset.jsf import *
from utilities.argument_funcs import parse_train_args, parse_eval_args
import torch.nn as nn
from thop import profile
# torch.set_printoptions(profile="full")

# train_epoch


def params(dataloader, model, model_disc):

    args = parse_train_args()
    model.eval()
    for batch_num, batch in enumerate(dataloader):
        flops, params = profile(model.module, (batch[0][0][0].cuda(args.gpu[0]),
                                               batch[0][0][1].cuda(args.gpu[0]),
                                               batch[0][0][2].cuda(args.gpu[0]))
                                )
        print('flops:', flops, 'params:', params)
        break


def train_with_adv(cur_epoch, model, model_disc, dataloader, loss, opt, opt_disc,
                lr_scheduler=None, lr_disc_scheduler=None, print_modulus=1):

    args = parse_train_args()
    out = -1
    start_epoch = 5
    model.train()
    model_disc.train()
    for batch_num, batch in enumerate(dataloader):
        time_before = time.time()

        opt.zero_grad()
        opt_disc.zero_grad()
        x = batch[0]
        tgt = batch[1]
        for i in range(len(batch[0])):
            if args.gpu[0] != -1:
                if isinstance(x[i], list):
                    for j in range(len(x[i])):
                        x[i][j] = x[i][j].cuda(device=args.gpu[0])
                if isinstance(x[i], torch.Tensor):
                    x[i] = x[i].cuda(device=args.gpu[0])

                if isinstance(tgt[i], list):
                    for j in range(len(tgt[i])):
                        tgt[i][j] = tgt[i][j].cuda(device=args.gpu[0])
                if isinstance(tgt[i], torch.Tensor):
                    tgt[i] = tgt[i].cuda(device=args.gpu[0])
            else:
                if isinstance(x[i], list):
                    for j in range(len(x[i])):
                        x[i][j] = x[i][j].cpu()
                        tgt[i][j] = tgt[i][j].cpu()
        tgt = tgt[0][0]
        tgt = tgt.flatten()
        with torch.no_grad():
            y1 = model.module(x[1][0], x[1][1], x[1][2])
            y1 = y1.reshape(y1.shape[0] * y1.shape[1], -1)
            loss1 = loss.forward(y1, tgt)
        y2 = model.module(x[0][0], x[0][1], x[0][2])
        # discriminator model loss:
        if args.gpu[0] != -1:
            real_disc_label = torch.ones(len(batch[0]), batch[1][0][0].shape[1], 1).to(args.gpu[0])
            fake_disc_label = torch.zeros(len(batch[0]), y2.shape[1], 1).to(args.gpu[0])
        else:
            real_disc_label = torch.ones(len(batch[0]), batch[1][0][0].shape[1], 1)
            fake_disc_label = torch.zeros(len(batch[0]), y2.shape[1], 1)

        softmax = nn.Softmax(dim=-1)
        d_fake_loss, d_fake_logits = model_disc(torch.argmax(softmax(y2), dim=-1), fake_disc_label)
        d_real_loss, d_real_logits = model_disc(batch[1][0][0], real_disc_label)
        loss3 = d_fake_loss + d_real_loss
        # y3 = model(x[2])
        # train for only CT
        # y = model(x)

        y2 = y2.reshape(y2.shape[0] * y2.shape[1], -1)
        loss2 = loss.forward(y2, tgt)
        # tgt = tgt.flatten()
        # add scheduled sampling
        # out = loss.forward(y, tgt)

        # out = loss3
        out = args.loss[0] * loss1 + args.loss[1] * loss2 + args.loss[2] * loss3

        out.backward()
        opt.step()
        opt_disc.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        if lr_disc_scheduler is not None:
            lr_disc_scheduler.step()

        time_after = time.time()
        time_took = time_after - time_before

        if (batch_num + 1) % print_modulus == 0:
            print("Epoch", cur_epoch, " Batch", batch_num + 1, "/", len(dataloader), "LR:", get_lr(opt_disc),
                  "Train total loss:", float(out), "Train loss1:", float(loss1), "Train loss2:", float(loss2),
                  "Train loss3:", float(loss3), "Time (s):", time_took)

    return


# train_epoch
def train_epoch(cur_epoch, model, dataloader, loss, opt, lr_scheduler=None, print_modulus=1):

    args = parse_train_args()

    model.train()
    for batch_num, batch in enumerate(dataloader):
        time_before = time.time()

        opt.zero_grad()
        x = batch[0]
        tgt = batch[1]
        for i in range(len(batch[0])):
            if args.gpu[0] != -1:
                if isinstance(x[i], list):
                    for j in range(len(x[i])):
                        x[i][j] = x[i][j].cuda(device=args.gpu[0])
                if isinstance(x[i], torch.Tensor):
                    x[i] = x[i].cuda(device=args.gpu[0])

                if isinstance(tgt[i], list):
                    for j in range(len(tgt[i])):
                        tgt[i][j] = tgt[i][j].cuda(device=args.gpu[0])
                if isinstance(tgt[i], torch.Tensor):
                    tgt[i] = tgt[i].cuda(device=args.gpu[0])
            else:
                if isinstance(x[i], list):
                    for j in range(len(x[i])):
                        x[i][j] = x[i][j].cpu()
                        tgt[i][j] = tgt[i][j].cpu()
        tgt = tgt[0][0]
        tgt = tgt.flatten()
        # with torch.no_grad():
        #     y1 = model(x[1])
        #     y1 = y1.reshape(y1.shape[0] * y1.shape[1], -1)
        #     loss1 = loss.forward(y1, tgt)
        y2 = model(x[0])
        # y3 = model(x[2])
        # train for only CT
        # y = model(x)

        y2 = y2.reshape(y2.shape[0] * y2.shape[1], -1)
        loss2 = loss.forward(y2, tgt)
        # tgt = tgt.flatten()
        # add scheduled sampling
        # out = loss.forward(y, tgt)

        loss2.backward()
        # out = args.loss[0] * loss1 + args.loss[1] * loss2

        opt.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        time_after = time.time()
        time_took = time_after - time_before

        if (batch_num + 1) % print_modulus == 0:
            print("Epoch", cur_epoch, " Batch", batch_num + 1, "/", len(dataloader), "LR:", get_lr(opt),
                  "Train total loss:", float(loss2),
                  "Time (s):", time_took)

    return


def train_with_pure_transformer(cur_epoch, model, dataloader, loss, opt, lr_scheduler=None, print_modulus=1):

    args = parse_train_args()

    model.train()
    for batch_num, batch in enumerate(dataloader):
        time_before = time.time()

        opt.zero_grad()

        x = batch[0][0][0].to(args.gpu[0])
        tgt = batch[1][0][0].to(args.gpu[0])

        y = model(x)

        y = y.reshape(y.shape[0] * y.shape[1], -1)
        tgt = tgt.flatten()

        out = loss.forward(y, tgt)

        out.backward()
        opt.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        time_after = time.time()
        time_took = time_after - time_before

        if (batch_num + 1) % print_modulus == 0:
            print("Epoch", cur_epoch, " Batch", batch_num + 1, "/", len(dataloader), "LR:", get_lr(opt),
                  "Train loss:", float(out), "Time (s):", time_took)

    return


# eval_model
def eval_model(model, dataloader, loss):

    model.eval()
    args = parse_train_args()
    avg_acc = -1
    avg_loss = -1
    with torch.set_grad_enabled(False):
        n_test = len(dataloader)
        sum_loss = 0.0
        sum_acc = 0.0
        for batch in tqdm.tqdm(dataloader):
            x = batch[0]
            tgt = batch[1]
            for i in range(len(batch[0])):
                if args.gpu[0] != -1:
                    if isinstance(x[i], list):
                        for j in range(len(x[i])):
                            x[i][j] = x[i][j].cuda(device=args.gpu[0])
                    if isinstance(x[i], torch.Tensor):
                        x[i] = x[i].cuda(device=args.gpu[0])

                    if isinstance(tgt[i], list):
                        for j in range(len(tgt[i])):
                            tgt[i][j] = tgt[i][j].cuda(device=args.gpu[0])
                    if isinstance(tgt[i], torch.Tensor):
                        tgt[i] = tgt[i].cuda(device=args.gpu[0])
                else:
                    x[i] = x[i].cpu()
                    tgt[i] = tgt[i].cpu()
            tgt = tgt[0][0]
            tgt = tgt.flatten()

            # with torch.no_grad():
            #     y1 = model(x[0])
            #     y1 = y1.reshape(y1.shape[0] * y1.shape[1], -1)
            #     loss1 = loss.forward(y1, tgt)
            y2 = model.module(x[0][0], x[0][1], x[0][2])
            y2 = y2.reshape(y2.shape[0] * y2.shape[1], -1)
            loss2 = loss.forward(y2, tgt)
            out = loss2

            sum_acc += float(compute_jsf_accuracy(y2, tgt))

            # y = y.reshape(y.shape[0] * y.shape[1], -1)
            # tgt = tgt.flatten()

            # out = loss.forward(y, tgt)

            sum_loss += float(out)

        avg_loss = sum_loss / n_test
        avg_acc = sum_acc / n_test

    return avg_loss, avg_acc


# eval_model
def get_metrics(model, dataloader):
    """
    Calculate TER: token error rate
    """
    args = parse_eval_args()
    model.eval()
    # TER
    with torch.set_grad_enabled(False):
        n_test = len(dataloader)
        c_acc, Ns_acc, Bs_acc, Na_acc, Ba_acc, Nt_acc, Bt_acc, Nb_acc, Bb_acc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ter = []
        for batch in tqdm.tqdm(dataloader):
            x = batch[0]
            tgt = batch[1]
            for i in range(len(batch[0])):
                if args.gpu[0] != -1:
                    if isinstance(x[i], list):
                        for j in range(len(x[i])):
                            x[i][j] = x[i][j].cuda(device=args.gpu[0])
                    if isinstance(x[i], torch.Tensor):
                        x[i] = x[i].cuda(device=args.gpu[0])

                    if isinstance(tgt[i], list):
                        for j in range(len(tgt[i])):
                            tgt[i][j] = tgt[i][j].cuda(device=args.gpu[0])
                    if isinstance(tgt[i], torch.Tensor):
                        tgt[i] = tgt[i].cuda(device=args.gpu[0])
                else:
                    if isinstance(x[i], list):
                        for j in range(len(x[i])):
                            x[i][j] = x[i][j].cpu()
                            tgt[i][j] = tgt[i][j].cpu()

            y = model.module(x[0][0], x[0][1], x[0][2])
            # TER
            ter.append(compute_jsf_ter(y, tgt))

        for i in ter:
            c_acc += i[0]
            Bs_acc += i[1]
            Ns_acc += i[2]
            Ba_acc += i[3]
            Na_acc += i[4]
            Bt_acc += i[5]
            Nt_acc += i[6]
            Bb_acc += i[7]
            Nb_acc += i[8]
        TER = [c_acc / n_test, Bs_acc / n_test, Ns_acc / n_test, Ba_acc / n_test, Na_acc / n_test,
               Bt_acc / n_test, Nt_acc / n_test, Bb_acc / n_test, Nb_acc / n_test]
    # clear nan , or np.mean will only be nan if one is nan
    return TER


