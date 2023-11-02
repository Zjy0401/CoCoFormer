import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader

from collections import OrderedDict

from dataset.jsf import create_jsf_datasets
from model.CoCoFormer import CoCoformer

from utilities.constants import *
from utilities.device import get_device, use_cuda
from utilities.argument_funcs import parse_eval_args, print_eval_args
from utilities.run_model import eval_model, get_metrics



# main


def main():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Entry point. Evaluates a model specified by command line arguments
    ----------
    """

    args = parse_eval_args()
    print_eval_args(args)

    if(args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")
        print("")

    # Test dataset
    _, test_dataset = create_jsf_datasets(args.dataset_dir, args.max_sequence)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_workers)
    f = open(args.word2event, 'rb')
    word2event = pickle.load(f)
    # reverse the vector event2word
    event2word = {}
    for key, val in word2event.items():
        event2word[val] = key
    model = CoCoformer(n_layers=args.n_layers, num_heads=args.num_heads,
                             d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                             max_sequence=args.max_sequence, rpr=args.rpr, word2event=word2event,
                             event2word=event2word)

    # model = MusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
    #             d_model=args.d_model, dim_feedforward=args.dim_feedforward,
    #             max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())
    if args.gpu[0] != -1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu)
        model = model.cuda(device=args.gpu[0])
    # for k, v in model.state_dict().items():
    #     print(k, ' :', v.shape)
    # state_dict = torch.load(args.model_weights)
    # for k, v in state_dict.items():
    #     print(k, ' :', v.shape)
    # TODO: some params is unnecessary, but need to check every paras of layers, only size=1 layers is unnecessary!
    model.load_state_dict(torch.load(args.model_weights, map_location=lambda storage, loc: storage.cuda(device=args.gpu[0])), strict=False)

    # No smoothed loss
    loss = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)

    print("Evaluating:")
    model.eval()
    # rhy_consistency, harmonic_consistency, \
    #     vocal_part_entropy, vocal_part_cross_entropy = get_my_metrics(model, test_loader, loss)
    avg_loss, avg_acc, = eval_model(model, test_loader, loss)
    TER = get_metrics(model, test_loader)

    print("Avg loss:", avg_loss, "Avg acc:", avg_acc)
    print("C_acc:", TER[0],
          "BS_acc:", TER[1],
          "NS_acc:", TER[2],
          "BA_acc:", TER[3],
          "NA_acc:", TER[4],
          "BT_acc:", TER[5],
          "NT_acc:", TER[6],
          "BB_acc:", TER[7],
          "NB_acc:", TER[8])
    print(SEPERATOR)
    # print("rhy consistency:", rhy_consistency)
    # print("harmonic_consistency:", harmonic_consistency)
    # print("vocal_part_entropy:", vocal_part_entropy)
    # print("vocal_part_cross_entropy:", vocal_part_cross_entropy)
    print("")


if __name__ == "__main__":
    main()
