import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F


"""
data augmentation util
"""
# add noise
def gaussian_noise(source, noise_base=0.1, dtype=torch.float32):
    x = noise_base + torch.zeros_like(source, dtype=dtype, device=source.device)
    noise = torch.normal(mean=torch.tensor([0.0]).to(source.device), std=x).to(source.device)
    return noise

def add_gaussian_noise(source, noise_base=0.1, dtype=torch.float32):
    noise = gaussian_noise(source, noise_base, dtype)
    source = source + noise
    return source

def mul_gaussian_noise(source, noise_base=0.1, dtype=torch.float32):
    noise = gaussian_noise(source, noise_base, dtype)
    source = torch.mul(source, 1 + noise)
    return source


# representation
def representation(config, count_dict, interaction, embeddings):
    if not config['open_represent']:
        return

    if count_dict['idx'] and not (count_dict['idx'] % 20 == 0 or count_dict['idx'] == 2):
        count_dict['idx'] += 1
        return
    else:
        count_dict['idx'] += 1

    if not config['rep_index']:
        config['rep_index'] = 0

    counts = interaction['iu_count']
    path = r'{}/{}_{}.png'.format(config['represent_path'], config['dataset'], str(config['rep_index']))
    # begin
    counts = counts * 100 / counts.max()

    xs1, ys1 = [], []
    xs2, ys2 = [], []
    xs3, ys3 = [], []
    xs4, ys4 = [], []

    for idx in range(embeddings.shape[0]):
        # i = pos_item[idx]
        in_ = counts[idx]
        i_e = embeddings[idx]
        i_e = i_e.view(2, -1).detach()
        if in_ <= 25:
            xs1.extend(i_e[0].cpu().numpy().tolist())
            ys1.extend(i_e[1].cpu().numpy().tolist())
        elif in_ <= 50:
            xs2.extend(i_e[0].cpu().numpy().tolist())
            ys2.extend(i_e[1].cpu().numpy().tolist())
        elif in_ <= 75:
            xs3.extend(i_e[0].cpu().numpy().tolist())
            ys3.extend(i_e[1].cpu().numpy().tolist())
        elif in_ <= 100:
            xs4.extend(i_e[0].cpu().numpy().tolist())
            ys4.extend(i_e[1].cpu().numpy().tolist())

    g = sns.JointGrid()
    sns.set_style("darkgrid")  # darkgrid, whitegrid, dark, white, ticks
    # sns.set_context("notebook")
    sns.set_context("poster", font_scale=1.5, rc={"lines.linewidth": 3})

    color = '#fde725'  # yellow
    # df = pd.DataFrame({"x": xs1, "y": ys1})
    sns.scatterplot(x=xs1, y=ys1, ax=g.ax_joint, color=color)
    sns.kdeplot(x=xs1, ax=g.ax_marg_x, color=color)
    sns.kdeplot(y=ys1, ax=g.ax_marg_y, color=color)

    color = '#35b779'  # green
    sns.scatterplot(x=xs2, y=ys2, ax=g.ax_joint, color=color)
    sns.kdeplot(x=xs2, ax=g.ax_marg_x, color=color)
    sns.kdeplot(y=ys2, ax=g.ax_marg_y, color=color)

    color = '#440154'  # violet
    sns.scatterplot(x=xs3, y=ys3, ax=g.ax_joint, color=color)
    sns.kdeplot(x=xs3, ax=g.ax_marg_x, color=color)
    sns.kdeplot(y=ys3, ax=g.ax_marg_y, color=color)

    color = '#31688e'  # blue
    sns.scatterplot(x=xs4, y=ys4, ax=g.ax_joint, color=color)
    sns.kdeplot(x=xs4, ax=g.ax_marg_x, color=color)
    sns.kdeplot(y=ys4, ax=g.ax_marg_y, color=color)

    # plt.colorbar()
    g.set_axis_labels('', '')
    plt.savefig(path)
    plt.close()

    config['rep_index'] += 1

def node_dropout(seq, seq_len, dropout_rate=0.1):
    r"""Randomly discard some points.
    """
    seq_len = torch.clone(seq_len) # clone
    mask = torch.rand([seq.shape[0],seq.shape[1]]).to(seq.device) >= dropout_rate
    mask[:, 0] = True
    seq1 = torch.mul(seq, mask)
    arr = seq1.tolist()

    for i in range(len(arr)):
        row = arr[i]
        lens = seq_len[i]
        for j in range(lens-1):
            # first value is skipped
            j += 1
            item = row[j]
            if item == 0:
                lens -= 1
                del row[j]
                row.append(0)
    seq = torch.tensor(arr).to(seq.device)
    return seq, seq_len

def calc_similarity_batch(a, b):
    representations = torch.cat([a, b], dim=0)
    return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

def cos_sim(z1: torch.Tensor, z2: torch.Tensor):
    '''
    cos similarity
    '''
    z1, z2 = z1.unsqueeze(1), z2.unsqueeze(0)
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    sim = torch.mm(z1, z2.t())
    return sim

def sim(z1: torch.Tensor, z2: torch.Tensor):
    '''
    cos similarity
    '''
    z = torch.cat((z1, z2), dim=0)  # 2B * D
    sim = torch.mm(z, z.T)  # 2B * 2B
    return sim

def cal_item_x_y_count(config, item_seq, item_seq_len, method_arr):
    item_x_y_count_path = r'{}/{}_item_x_y_count.npy'.format(config['checkpoint_dir'], config['dataset'])
    if os.path.exists(item_x_y_count_path):
        item_x_y_count = np.load(item_x_y_count_path)
    else:
        # x_y_count = {x_y: count}
        item_x_y_count = {}
        for i in range(len(item_seq)):
            seq = item_seq[i]
            seq_len = item_seq_len[i]
            if seq_len <= 1:
                continue

            for i1 in range(seq_len):
                x = seq[i1]
                for i2 in range(seq_len - 1):
                    y = seq[i2 + 1]
                    if x == y:
                        continue
                    key = (str(x.item()) + '_' + str(y.item())) if y < x else (str(y.item()) + '_' + str(x.item()))
                    if item_x_y_count.__contains__(key):
                        item_x_y_count[key] += 1
                    else:
                        item_x_y_count[key] = 1

        # sort by count(r) reverse
        item_x_y_count = np.array(sorted(item_x_y_count.items(), key=lambda d: d[1], reverse=True))
        np.save(item_x_y_count_path, item_x_y_count)

    # item_x_y_count
    # train_data.dataset.item_x_y_count = item_x_y_count
    # print(item_x_y_count)
    replace_rate = float(method_arr[2])
    item_x_y_count_mini = item_x_y_count[0:int(len(item_x_y_count) * replace_rate)]
    item_x_y_dict = {}
    for i in item_x_y_count_mini:
        key = int(i[0].split('_')[0])
        val = int(i[0].split('_')[1])
        if item_x_y_dict.__contains__(key):
            item_x_y_dict[key].append(val)
        else:
            item_x_y_dict[key] = [val]
    return item_x_y_dict

def joint_result(test_result):
    str_ = None
    for key in test_result:
        if str.startswith(key, 'recall@') or str.startswith(key, 'ndcg@') or str.startswith(key, 'hit@'):
            if str_ == None:
                str_ = ''
            else:
                str_ += '	'
            str_ += str(test_result[key])
    return str_

def joint_config(config):
    config_ = 'model:' + str(config['model']) +',dataset:' + str(config['dataset'])
    if config['batch_size']:
        config_ += ',batch_size:' + str(config['batch_size'])
    if config['loss_func_temp']:
        config_ += ',loss_func:' + str(config['loss_func_temp'])
    if config['open_cl']:
        config_ += ',open_cl:' + str(config['open_cl'])
    if config['data_aug_method']:
        config_ += ',data_aug:' + str(config['data_aug_method'])
    if config['attn_dropout_prob']:
        config_ += ',dropout:' + str(config['attn_dropout_prob'])
    if config['hidden_size']:
        config_ += ',hidden_size:' + str(config['hidden_size'])
    if config['phi']:
        config_ += ',phi:' + str(config['phi'])
    if config['tf_weight']:
        config_ += ',tf_weight:' + str(config['tf_weight'])
    if config['gnn_weight']:
        config_ += ',gnn_weight:' + str(config['gnn_weight'])
    if config['reg_weight']:
        config_ += ',reg_weight:' + str(config['reg_weight'])
    if config['lambda1']:
        config_ += ',lambda1:' + str(config['lambda1'])
    if config['negative_sample_batch']:
        config_ += ',negative_sample_batch:' + str(config['negative_sample_batch'])
    if config['nd_rate']:
        config_ += ',nd_rate:' + str(config['nd_rate'])
    if config['noise_base']:
        config_ += ',noise_base:' + str(config['noise_base'])
    if config['pgd']:
        config_ += ',pgd:' + str(config['pgd'])
    if config['noise_grad_base']:
        config_ += ',noise_grad_base:' + str(config['noise_grad_base'])

    return config_

def cal_item_seq_count(config, item_seq, item_seq_len, method_arr):
    item_seq_dict = None
    return item_seq_dict

def rand_sample(high, size=None, replace=True):
    r"""Randomly discard some points or edges.
    Args:
        high (int): Upper limit of index value
        size (int): Array size after sampling
    Returns:
        numpy.ndarray: Array index after sampling, shape: [size]
    """
    a = np.arange(high)
    sample = np.random.choice(a, size=size, replace=replace)
    return sample

def cal_alignment_and_uniformity(z_i, z_j, origin_z, batch_size):
    """
    We do not sample negative examples explicitly.
    Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
    """
    N = 2 * batch_size

    z = torch.cat((z_i, z_j), dim=0)

    # pairwise l2 distace
    sim = torch.cdist(z, z, p=2)

    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)

    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    alignment = positive_samples.mean()

    # pairwise l2 distace
    sim = torch.cdist(origin_z, origin_z, p=2)
    mask = torch.ones((batch_size, batch_size), dtype=bool)
    mask = mask.fill_diagonal_(0)
    negative_samples = sim[mask].reshape(batch_size, -1)
    uniformity = torch.log(torch.exp(-2 * negative_samples).mean())

    return alignment, uniformity