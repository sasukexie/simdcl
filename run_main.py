import argparse
from recbole.quick_start import run_recbole as run_recbole
import uuid
import setproctitle
setproctitle.setproctitle("PROC@MAIN")


def main(model_name, dataset_name, parameter_dict, config_file=None):
    # 1.set param
    parser = argparse.ArgumentParser()
    # set model
    parser.add_argument('--model', '-m', type=str, default=model_name, help='name of models')
    # set datasets # ml-1m,ml-20m,amazon-books,lfm1b-tracks
    parser.add_argument('--dataset', '-d', type=str, default=dataset_name, help='name of datasets')
    # set config
    parser.add_argument('--config_files', type=str, default=None, help='config files')
    # get param
    args, _ = parser.parse_known_args()
    # config list
    yaml_dict = {'base': 'zone/baseline.yaml', 'main': 'zone/main.yaml'}

    if config_file:
        config_file = yaml_dict[config_file]
    else:
        if model_name == 'SimDCL':
            config_file = yaml_dict['main']

    parameter_dict['running_flag'] = str(uuid.uuid4())
    print("running_flag: ", parameter_dict['running_flag'])
    print("config_file: ", config_file)
    # 2.call recbole_trm: config,dataset,model,trainer,training,evaluation
    if config_file:
        config_file_list = [config_file]
        run_recbole(model=args.model, dataset=args.dataset, config_file_list=config_file_list, config_dict=parameter_dict)
    else:
        run_recbole(model=args.model, dataset=args.dataset, config_dict=parameter_dict)


def tranfer_dict(parameter_dict,parameter_dict1):
    if parameter_dict1:
        for key in parameter_dict1.keys():
            parameter_dict[key] = parameter_dict1[key]
    return parameter_dict

def process_base():
    # param
    # config_file = None  # None/base
    config_file = 'base'
    parameter_dict = {
        'epochs': 100,
        'train_batch_size': 1024,
    }

    # set model # MODEL,SASCTS,SASRec,BERT4Rec,BPR,GRU4RecF
    # set datasets # steam,Amazon_Sports_and_Outdoors,Amazon_All_Beauty,ml-1m,Amazon_Software,Amazon_Books
    model_name_arr = ['BERT4Rec']  # 'BPR','LightGCN','S3Rec','BERT4Rec','SASRec','GRU4Rec','Caser'
    dataset_name_arr = ['steam','yelp','lfm1b-tracks','ml-1m']
    for model_name in model_name_arr:
        for dataset_name in dataset_name_arr:
            main(model_name, dataset_name, parameter_dict, config_file)


def process_0(parameter_dict):
    # param
    # set model # MODEL,SimDCL,SASRec,BERT4Rec,BPR,GRU4RecF
    # set datasets # ml-1m,ml-20m,Amazon_Books,Amazon_Sports_and_Outdoors,Amazon_All_Beauty,amazon-books,lfm1b-tracks
    model_name_arr = ['SimDCL']  # GRU4RecF,BPR
    dataset_name_arr = ['steam']  # ] #
    for model_name in model_name_arr:
        for dataset_name in dataset_name_arr:
            main(model_name, dataset_name, parameter_dict)


def process_1(parameter_dict):
    # param
    # set model
    config_file = 'main'
    parameter_dict1 = {
        ############# common ##############
        'epochs': 100,
        'train_batch_size': 512,  # 4096,1024
        'eval_batch_size': 256,
        'topk': [1, 5, 10, 20],
        'metrics': ["Recall", "NDCG", "Hit", "MRR", "Precision"],
        'valid_metric': 'Recall@5',# MRR@10,NDCG@10,Recall@5,Recall@1
        'MAX_ITEM_LIST_LENGTH': 50,
        # 'max_user_inter_num': 50,
        # 'data_path': "./dataset/noise/",
        ############# model ##############
        # steam.t0.5, loss1=cts1; loss2=debias
        't_weight': 0.5,
        'g_weight': 0.5,
        'reg_weight': 1e-5,
        'open_cl': False,
        'open_reg': True,
        'loss_func_temp': ['loss_1#1.0'],
        # loss_func#temp: ['loss_1#1.0'],['loss_2#1.0']
        # em:gn,negative; node:self,nd,gn,negative; ne:nd#gn; tg:gn,gn_m
        'data_aug_method': 'tg:gn',
        'nd_rate': 0.1,
        'noise_base': 0.1,  # 0.1,0.01,0.001,0.0001,0.00001,0.000001
        'negative_noise_base_arr': [0.1], #[0.1,0.01,0.001,0.0001] # DA: negative
        'pgd': 1,  # weight for PGD turns. #loss2 # ml-1m 0
        'phi': 0.85,  # false negative
        'sample_batch': 3,  # negative sample batch, loss2
        # 'loss_type': 'CE',
        # represent
        'open_represent': False,
        # uniform & align
        'open_ali_uni': True,
        'lambda1': 0.01, # 0.01 loss*lambda1,
    }
    tranfer_dict(parameter_dict, parameter_dict1)

    model_name_arr = ['SimDCL']  # SimDCL,SASRec,BERT4Rec,BPR,GRU4RecF,BPR
    dataset_name_arr = ['steam','yelp'] # ['steam','yelp','lfm1b-tracks','ml-1m']

    for model_name in model_name_arr:
        for dataset_name in dataset_name_arr:
            # data aug
            noise_base_arr = [1e-1]  # [1e-1,1e-2,1e-3,1e-4,1e-5]

            # 0.1,steam_0.6
            if ['steam'].__contains__(dataset_name):
                dropout_prob_arr = [0.5]
            elif ['lfm1b-tracks','ml-1m'].__contains__(dataset_name):
                dropout_prob_arr = [0.1]
            elif ['yelp'].__contains__(dataset_name):
                dropout_prob_arr = [0.7]
                noise_base_arr = [1e-3]
            else:
                dropout_prob_arr = [0.5]
            if ['steam'].__contains__(dataset_name):
                parameter_dict['open_reg'] = False
            else:
                parameter_dict['open_reg'] = True

            # dropout_prob_arr = [0.1,0.3,0.5,0.7,0.8]
            hidden_size_arr = [64]  # 64,128
            nd_rate_arr = [0.1]  # [0.1,0.2,0.3,0.4,0.5] # da
            phi_arr = [0.85]  # [0.98,0.95,0.85,0.75,0.65,0.55,0.45,0.35,0.25,0.15,0.05]
            # em:gn,negative; node:self,nd,gn,negative; ne:nd#gn; tg:gn,gn_m
            data_aug_method_arr = ['tg:gn']  # ['em:gn','em:gn_m','em:an','ne:nd#an']
            # cl
            loss_func_temp_arr = [['loss_1#1.0']]  # 'loss_1#1.0','loss_2#1.0','loss_3#1.0','loss_4#1.0'
            sample_batch_arr = [2]  # [2,3],loss_3/6, ml-1m:1

            for loss_func_temp in loss_func_temp_arr:
                for data_aug_method in data_aug_method_arr:
                    for dropout_prob in dropout_prob_arr:
                        for hidden_size in hidden_size_arr:
                            for noise_base in noise_base_arr:
                                for nd_rate in nd_rate_arr:
                                    parameter_dict['nd_rate'] = nd_rate
                                    parameter_dict['noise_base'] = noise_base
                                    parameter_dict['loss_func_temp'] = loss_func_temp
                                    parameter_dict['data_aug_method'] = data_aug_method
                                    parameter_dict['attn_dropout_prob'] = dropout_prob
                                    parameter_dict['hidden_dropout_prob'] = dropout_prob
                                    parameter_dict['hidden_size'] = hidden_size
                                    parameter_dict['inner_size'] = hidden_size * 4
                                    if str(loss_func_temp).__contains__('loss_1') or str(loss_func_temp).__contains__('loss_4'):
                                        for phi in phi_arr:
                                            for sample_batch in sample_batch_arr:
                                                parameter_dict['phi'] = phi
                                                parameter_dict['sample_batch'] = sample_batch
                                                main(model_name, dataset_name, parameter_dict, config_file)
                                    else:
                                        main(model_name, dataset_name, parameter_dict, config_file)


if __name__ == '__main__':
    parameter_dict = {
        'seed': 2021,
        'epochs': 100,
        'train_batch_size': 4096,  # 4096,1520
        'metrics': ["Recall", "NDCG", "Hit", "MRR", "Precision"],  # HR/Hit: Hit Ratio
        'topk': [5, 10, 20],
        'fast_sample_eval': 1,
        # 'min_item_inter_num': 5,
        # steam.t0.5, loss1=cts; loss2=debias
        'loss_func_temp': ['loss_1#1.0'],  # loss_func#temp: ['loss_1#1.0'],['loss_2#1.0'],['loss_3#1.0'],['loss_1#1.0','loss_2#1.0']
        # em:gn,negative; node:self,nd,gn,negative; ne:nd#gn; tg:gn,gn_m
        'data_aug_method': 'em:gn',
        'nd_rate': 0.1,
        'noise_base': 0.01,  # 0.1,0.01,0.001,0.0001,0.00001,0.000001
        'negative_noise_base_arr': [0.1],  # 0.1,0.01,0.001,0.0001,0.00001,0.000001
        'pgd': 5, # weight for PGD turns.
        'phi': 0.85, # false negative
        'sample_batch': 3, # negative sample batch, loss4
        # represent
        'open_represent': False,
        'represent_path': '/media/data/temp/SimDCL',
        # uniform & align
        'open_ali_uni': False,
        'align_w': 1,  # Alignment loss weight
        'unif_w': 1,  # Uniformity loss weight
        'align_alpha': 2,  # alpha in alignment loss
        'unif_t': 2,  # t in uniformity loss
    }
    # param
    # set model # MODEL,SimDCL,SASRec,BERT4Rec,BPR,GRU4RecF
    # set datasets # ['steam','lfm1b-tracks','ml-1m']
    # process_base(parameter_dict)
    process_1(parameter_dict)


