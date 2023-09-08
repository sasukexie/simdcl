# @Time   : 2020/10/6
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

"""
recbole.quick_start
########################
"""
import logging
from logging import getLogger

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_model, get_trainer, init_seed
from recbole.utils.utils import set_color
import common.utils.tool as tool

def run_recbole(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str): model name
        dataset (str): dataset name
        config_file_list (list): config files used to modify experiment parameters
        config_dict (dict): parameters dictionary used to modify experiment parameters
        saved (bool): whether to save the model
    """
    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    # init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    import os
    log_dir = os.path.dirname(logger.handlers[0].baseFilename)
    config['log_dir'] = log_dir
    
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = get_model(config['model'])(config, train_data).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )

    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.decomposition import TruncatedSVD

    embedding_matrix = model.item_embedding.weight[1:].cpu().detach().numpy()
    svd = TruncatedSVD(n_components=2)
    svd.fit(embedding_matrix)
    comp_tr = np.transpose(svd.components_)
    proj = np.dot(embedding_matrix, comp_tr)
    
    cnt = {}
    for i in dataset['item_id']:
        if i.item() in cnt:
            cnt[i.item()] += 1
        else:
            cnt[i.item()] = 1
    
    freq = np.zeros(embedding_matrix.shape[0])
    for i in cnt:
        freq[i-1] = cnt[i]
    
    # freq /= freq.max()

    sns.set(style='darkgrid')
    sns.set_context("notebook", font_scale=1.8, rc={"lines.linewidth": 3, 'lines.markersize': 20})
    plt.figure(figsize=(6, 4.5))
    plt.scatter(proj[:, 0], proj[:, 1], s=1, c=freq, cmap='viridis_r')
    plt.colorbar()
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    # plt.axis('square')
    # plt.show()
    # pig1: item embedding representation
    plt.savefig(log_dir + f'/{config["model"]}-{config["dataset"]}.pdf', format='pdf', transparent=False, bbox_inches='tight')
    
    from scipy.linalg import svdvals
    svs = svdvals(embedding_matrix)
    svs /= svs.max()
    np.save(log_dir + f'/{config["model"]}-{config["dataset"]}_svs.npy', svs)

    sns.set(style='darkgrid')
    sns.set_context("notebook", font_scale=1.8, rc={"lines.linewidth": 3, 'lines.markersize': 20})
    plt.figure(figsize=(6, 4.5))
    plt.plot(svs)
    # plt.show()
    # pig2: Model training matrix singular value SVD
    # The singular value of a matrix is an important concept in linear algebra. For an m×n matrix A, Singular Value Decomposition (SVD) is the decomposition of matrix A into the product of three matrices: A = UΣV^T, where U and V are orthogonal matrices and Σ is a diagonal matrix.
    # In singular value decomposition, the elements on the diagonal of Σ are called singular values of the matrix A. The singular values are arranged in order from greatest to smallest, indicating the importance of the matrix A in each dimension. The number of singular values is equal to the rank of matrix A.
    # Singular value decomposition is widely used in many fields, such as data dimensionality reduction, image compression, recommendation system, etc. It can extract the important features of the matrix, and the original matrix can be represented by an approximate matrix of lower dimensions, thus reducing the storage space and computational complexity.
    plt.savefig(log_dir + f'/{config["model"]}-{config["dataset"]}_svs.pdf', format='pdf', transparent=False, bbox_inches='tight')

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    # tools
    str_ = tool.joint_result(test_result)
    logger.info(set_color('test value', 'red') + f': {str_}')
    config_ = tool.joint_config(config)
    logger.info(set_color('config value', 'red')+ f': {config_}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def objective_function(config_dict=None, config_file_list=None, saved=True):
    r""" The default objective_function used in HyperTuning

    Args:
        config_dict (dict): parameters dictionary used to modify experiment parameters
        config_file_list (list): config files used to modify experiment parameters
        saved (bool): whether to save the model
    """

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = get_model(config['model'])(config, train_data).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False, saved=saved)
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }
