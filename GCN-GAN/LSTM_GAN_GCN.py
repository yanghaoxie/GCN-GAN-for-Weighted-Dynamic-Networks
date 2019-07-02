import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

# LSTM+GAN+GCN Model for temporal link predicton of weighted dynamic networks
# LSTM+GAN+GCN带权动态网络时序链路预测模型

def read_data(name_pre, time_index, node_num, max_thres):
    '''
    Function to read the network snapshot of specific time slice
    读取特定时间片下的网络快照的函数
    :param name_pre: the name prefix of the data file 数据文件名称前缀
    :param time_index: index of time slice 时间片索引
    :param node_num: number of nodes in the dynamic network 动态网络中节点总数
    :param max_thres: threshold of the maximum edge weight 最大边权值的阈值
    :return: adjacency matrix of the specific time slice 指定时间片的邻接矩阵
    '''
    print('Read network snapshot #%d'%(time_index))
    #Initialize the adjacency matrix 初始化邻接矩阵
    curAdj = np.mat(np.zeros((node_num, node_num)))
    #Read the network snapshot of current time slice 读取当前时间片的网络快照
    f = open('%s_%d.txt'%(name_pre, time_index))
    line = f.readline()
    while line:
        seq = line.split()
        #print(seq)
        src = int(seq[0]) #Index of the source node 源节点索引
        tar = int(seq[1]) #Index of the target node 目的节点索引
        seq[2] = float(seq[2])
        if seq[2]>max_thres:
            seq[2] = max_thres

        #Update the adjacency matrix 更新邻接矩阵
        curAdj[src, tar] = seq[2]
        curAdj[tar, src] = seq[2]
        line = f.readline()
    f.close()
    return curAdj

def var_init(m, n):
    '''
    Function to initialze the weight matrix
    初始化权重矩阵的函数
    :param m: number of rows of the weight matrix 权重矩阵的行数
    :param n: number of column of the weight matrix 权重矩阵的列数
    :return: the initialized weight matrix 初始化后的权重矩阵
    '''
    #in_dim = size[0]
    #w_stddev = 1. / tf.sqrt(in_dim / 2.)
    #return tf.random_normal(shape=size, stddev=w_stddev)
    init_range = np.sqrt(6.0 / (m+n))
    initial = tf.random_uniform([m, n], minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial)

def gen_noise(m, n):
    '''
    Function to generative noises with uniform discribution
    生成服从均匀分布噪声的函数
    :param m: number of rows of the noise matrix 噪声矩阵的行数
    :param n: number of columns of the noise matrix 噪声矩阵的列数
    :return: the noise matrix 噪声矩阵
    '''
    return np.random.uniform(0, 1., size=[m, n])
    #return np.random.normal(0.5, 1, [m, n])

def get_gcn_fact(adj):
    '''
    Function to calculate the GCN factor of a certain network snapshot
    计算某个网络快照GCN因子的函数
    :param adj: the adjacency matrix of a specific network snapshot 特定网络快照的邻接矩阵
    :return: the corresponding GCN factor 对应的GCN因子
    '''
    adj_ = adj + np.eye(node_num, node_num)
    row_sum = np.array(adj_.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.mat(np.diag(d_inv_sqrt))
    gcn_fact = d_mat_inv_sqrt*adj_*d_mat_inv_sqrt # The GCN factor GCN因子

    return gcn_fact

def get_noise_inputs():
    '''
    Function to construct the noise input list of the generaive network's GCN units
    构建生成网络GCN单元噪声输入列表的函数
    :return: the noise list 噪声列表
    '''
    # Construct the noise input list of the generative network 构建生成网络的噪声输入列表
    noise_inputs = []
    for i in range(window_size+1):
        noise_inputs.append(gen_noise(node_num, node_num))
    return noise_inputs

def gen_net(noise_input_phs, gcn_fact_phs):
    '''
    Function to define the generative network
    定义生成网络的函数
    :param noise_input_phs: list of the noise inputs 噪声输出列表
    :param gcn_fact_phs: list of the GCN factors GCN因子列表
    :return:
        gen_output: the output of the generative network 生成网络的输出
        LSTM_params: list of all the parameters in the LSTM hidden layer LSTM隐含层所有参数的列表
    '''
    # +--------------------+
    # GCN Input Layer -> LSTM Hidden Layer
    gcn_outputs = [] # Tensor list of the GCN Units' output GCN单元输出的张量列表
    for i in range(window_size+1):
        noise = noise_input_phs[i]
        gcn_fact = gcn_fact_phs[i]
        gcn_wei = gcn_weis[i]
        # +----------+
        # Conduct the GCN operation 执行GCN操作
        gcn_conv = tf.matmul(gcn_fact, noise)
        gcn_output = tf.sigmoid(tf.matmul(gcn_conv, gcn_wei))
        # +----------+
        # Reshape the output of current GCN unit 整理当前GCN单元的输出
        gcn_output = tf.reshape(gcn_output, [1, node_num*gen_hid_num0])
        # +----------+
        # Add current output to the tensor list 将当前输出加入张量列表
        gcn_outputs.append(gcn_output)
    # +--------------------+
    # LSTM Hidden Layer -> Output Layer
    LSTM_cells = rnn.MultiRNNCell([rnn.BasicLSTMCell(node_num*gen_hid_num0)]) #参数数量
    with tf.variable_scope("gen_net") as gen_net:
        LSTM_outputs, states = rnn.static_rnn(LSTM_cells, gcn_outputs, dtype=tf.float32)
        # Get the parameters of the generative network 获取生成网络的参数
        LSTM_params = [var for var in tf.global_variables() if var.name.startswith(gen_net.name)]
    # +--------------------+
    # Output Layer
    gen_output = tf.nn.sigmoid(tf.matmul(LSTM_outputs[-1], gen_output_wei) + gen_output_bias)

    return gen_output, LSTM_params

def disc_net(disc_input):
    '''
    Function to define the discriminative network
    定义判别网络的函数
    :param disc_input: the input of the discriminative network 判别网络的输入
    :return:
        disc_output: the output of the discriminative network 判别网络的输出
        disc_logit: the output of the output layer (without activation function) 判别网络输出层的输出(不考虑激活函数)
        disc_params: the parameters of the discriminative network 判别网络的参数
    '''
    # Input layer -> hidden layer #1
    disc_h1 = tf.nn.sigmoid(tf.matmul(disc_input, disc_wei1) + disc_bias1)
    # Hidden layer #1 -> Output layer
    disc_logit = tf.matmul(disc_h1, disc_wei2) + disc_bias2
    disc_output = tf.nn.sigmoid(disc_logit)

    return disc_output, disc_logit

def get_wei_KL(adj_est, gnd):
    '''
    Function to calculate the edge weight KL divergence
    :param adj_est: prediction result
    :param gnd: ground-truth
    :return: edge weight KL divergence
    '''
    sum_est = 0
    for r in range(node_num):
        for c in range(node_num):
            sum_est += adj_est[r, c]
    sum_gnd = 0
    for r in range(node_num):
        for c in range(node_num):
            sum_gnd += gnd[r, c]
    p = gnd/sum_gnd
    q = adj_est/sum_est
    edge_wei_KL = 0
    for r in range(node_num):
        for c in range(node_num):
            cur_KL = 0
            if q[r, c]>0 and p[r, c]>0:
                cur_KL = p[r, c]*np.log(p[r, c]/q[r, c])
            edge_wei_KL += cur_KL

    return edge_wei_KL

def get_mis_rate(adj_est, gnd):
    mis_sum = 0
    for r in range(node_num):
        for c in range(node_num):
            if (adj_est[r, c]>0 and gnd[r, c]==0) or (adj_est[r, c]==0 and gnd[r, c]>0):
                mis_sum += 1
    mis_rate = mis_sum/(node_num*node_num)

    return mis_rate

# +----------------------------------------+
# Set the parameters of the dynamic network
# 设置动态网络的参数
node_num = 38 # Number of nodes in the dynamic network 动态网络中节点总数
#node_num = 92
#+-----+
time_num = 1000 # Number of time slices 时间片总数
#time_num = 500
#+-----+
window_size = 10 # Window size of the history network snapshot to be considered 考虑的历史网络快照的窗口大小
# +-----------+
name_pre = ".\\data\\UCSB\\edge_list" # Prefix name of the data file 数据文件的前缀名称
#name_pre = ".\\data\\KAIST-HumMob\\edge_list"
#+----------+
max_thres = 2000 # Threshold of the maximum edge weight 最大边权值的阈值
#max_thres = 250

# +--------------------+
# Define the parameters of the nueral network
# 定义神经网络的参数
pre_epoch_num = 1000 # Number of pre-training epoches 预训练的带代数
epoch_num = 4000 # Number of training epoches 训练的迭代数
# 4000 (UCSB) 5000 (KAIST)
# +----------+
# Define the parameters of the generative network
# 定义生成网络的参数
gen_hid_num0 = 1
gen_hid_num1 = 64
# +-----+
# GCN Input Layer -> LSTM hideen Layer
gcn_weis = [] # List of the weighted matrixes of the GCN units GCN单元权重矩阵的列表
for i in range(window_size+1):
    gcn_weis.append(tf.Variable(var_init(node_num, gen_hid_num0)))
# +-----+
# LSTM Hidden Layer -> Output Layer
gen_output_wei = tf.Variable(var_init(node_num*gen_hid_num0, node_num*node_num))
gen_output_bias = tf.Variable(tf.zeros(shape=[node_num*node_num]))
# +-----+
# Parameter list of the generative network's output layer
# 生成网络输出层参数列表
gen_output_params = [gen_output_wei, gen_output_bias]
# +----------+
# Define the parameters of the discriminative network
# 定义判别网络的参数
disc_hid_num = 1024
# Input Layer -> Hidden Layer 1
disc_wei1 = tf.Variable(var_init(node_num*node_num, disc_hid_num))
disc_bias1 = tf.Variable(tf.zeros([disc_hid_num]))
# +-----+
# Hidden Layer 1 -> Output Layer
disc_wei2 = tf.Variable(var_init(disc_hid_num, 1))
disc_bias2 = tf.Variable(tf.zeros([1]))
# +-----+
# Parameter list of the discriminative network
# 判别网络的参数列表
disc_params = [disc_wei1, disc_bias1, disc_wei2, disc_bias2]
# +---------+
# Clipping bound for WGAN's traning process WGAN训练过程的限制边界
clip_ops = []
for var in disc_params:
    clip_bound = [-0.01, 0.01]
    clip_ops.append(
        tf.assign(var, tf.clip_by_value(var, clip_bound[0], clip_bound[1]))
    )
clip_disc_wei = tf.group(*clip_ops)

# +---------------------+
# Define the TF placeholders
# 定义TF占位符
gcn_fact_phs = [] # Placeholder list of the GCN factors GCN因子的占位符列表
noise_input_phs = [] # Placeholder list of the noise inpus 噪声输入的占位符列表
for i in range(window_size+1):
    gcn_fact_phs.append(tf.placeholder(tf.float32, shape=[node_num, node_num]))
    noise_input_phs.append(tf.placeholder(tf.float32, shape=[node_num, node_num]))
# +----------+
gnd_ph = tf.placeholder(tf.float32, shape=(1, node_num*node_num)) # Placeholder of the ground-truth 标准答案的占位符


# +---------------------+
# Construct the GAN
# 构建GAN
gen_output, LSTM_params = gen_net(noise_input_phs, gcn_fact_phs)
disc_real, disc_logit_real = disc_net(gnd_ph)
disc_fake, disc_logit_fake = disc_net(gen_output)

# +---------------------+
# Define the loss functin for the pre-train process of the generative network
# 定义生成网络预训练过程的损失函数
pre_gen_loss = tf.reduce_sum(tf.square(gnd_ph - gen_output))
# Difine the optimizer for the pre-train process of the generative network
# 定义生成网络预训练的优化器
#pre_gen_opt = tf.train.AdamOptimizer().minimize(pre_gen_loss, var_list=(gcn_weis+LSTM_params+gen_output_params))
pre_gen_opt = tf.train.RMSPropOptimizer(learning_rate=0.005).minimize(pre_gen_loss, var_list=(gcn_weis+LSTM_params+gen_output_params))
# 0.005 (UCSB)

# +--------------------+
# Define the loss function for GAN
# 定义GAN的损失函数
gen_loss = -tf.reduce_mean(disc_logit_fake)
disc_loss = tf.reduce_mean(disc_logit_fake) - tf.reduce_mean(disc_logit_real)

#disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1.-disc_fake))
#gen_loss = -tf.reduce_mean(tf.log(disc_fake))
# +----------------------+
#disc_loss_real = tf.reduce_mean(
#    tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_logit_real, labels=tf.ones_like(disc_logit_real)))
#disc_loss_fake = tf.reduce_mean(
#    tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_logit_fake, labels=tf.zeros_like(disc_logit_fake)))
#disc_loss = disc_loss_real + disc_loss_fake
#gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_logit_fake, labels=tf.ones_like(disc_logit_fake)))
# +------------------------+

# Define the optimizer for the generative network and the discriminative network
# 定义生成网络和判别网络的优化器
disc_opt = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(disc_loss, var_list=disc_params)
gen_opt = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss, var_list=(gcn_weis+LSTM_params+gen_output_params))
# 0.001, 0.001 (UCSB) 0.0005 0.0005 (KAIST)

#gen_opt = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_loss, var_list=(LSTM_params+gen_output_params))
#disc_opt = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_loss, var_list=disc_params)

#disc_opt = tf.train.AdamOptimizer().minimize(disc_loss, var_list=disc_params)
#gen_opt = tf.train.AdamOptimizer().minimize(gen_loss, var_list=(LSTM_params+gen_output_params))

# +--------------------+
avg_error = 0.0
avg_KL = 0.0
avg_mis = 0.0
cal_count = 0
# +--------------------+
# Run the neural network 运行神经网络
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for t in range(window_size, time_num-2):
    # Construct the GCN factor list of the generative network 构建生成网络的GCN因子列表
    gcn_facts = []
    for k in range(t-window_size, t+1):
        # Read and normalize the adjacency matrix 读取并归一化邻接矩阵
        adj = read_data(name_pre, k, node_num, max_thres)/max_thres
        gcn_fact = get_gcn_fact(adj)
        gcn_facts.append(gcn_fact)
    # +--------------------+
    # Construct the ground-truth vector 构建标准答案向量
    gnd = np.reshape(read_data(name_pre, t+1, node_num, max_thres ), (1, node_num*node_num))
    gnd /= max_thres

    # +----------------------+
    # Pretrain the generative network
    # 预训练生成网络
    loss_list = []
    for epoch in range(pre_epoch_num):
        # Construct the noise input list of the generative network 构建生成网络的噪声输入列表
        noise_inputs = get_noise_inputs()
        # +----------+
        # Construct the placeholder feed dictionary 构建占位符字典
        ph_dict = dict(zip(noise_input_phs, noise_inputs))
        ph_dict.update(dict(zip(gcn_fact_phs, gcn_facts)))
        ph_dict.update({gnd_ph: gnd})
        _, pre_g_loss, pre_g_output = sess.run([pre_gen_opt, pre_gen_loss, gen_output], feed_dict=ph_dict)
        loss_list.append(pre_g_loss)
        if epoch%100==0:
            print('Pre-Train #%d, G-Loss: %f'%(epoch, pre_g_loss))
        if epoch>500 and loss_list[epoch]>loss_list[epoch-1] and loss_list[epoch-1]>loss_list[epoch-2]:
            break

    # +----------------------+
    # Train the GAN
    # 训练GAN
    print('Train the GAN')
    for epoch in range(epoch_num):
        # Train the discriminative network
        # 训练判别网络
        # Construct the noise input list of the generative network 构建生成网络的噪声输入列表
        noise_inputs = get_noise_inputs()
        # +----------+
        # Construct the placeholder feed dictionary 构建占位符字典
        ph_dict = dict(zip(noise_input_phs, noise_inputs))
        ph_dict.update(dict(zip(gcn_fact_phs, gcn_facts)))
        ph_dict.update({gnd_ph : gnd})
        _, d_loss = sess.run([disc_opt, disc_loss], feed_dict=ph_dict)
        # +--------------------+
        # Train the generative network
        # 训练生成网络
        # Construct the noise input list of the generative network 构建生成网络的噪声输入列表
        noise_inputs = get_noise_inputs()
        # +----------+
        # Construct the placeholder feed dictionary 构建占位符字典
        ph_dict = dict(zip(noise_input_phs, noise_inputs))
        ph_dict.update(dict(zip(gcn_fact_phs, gcn_facts)))
        #ph_dict.update({gnd_ph: gnd})
        _, g_loss, g_output = sess.run([gen_opt, gen_loss, gen_output], feed_dict=ph_dict)
        # +----------+
        _ = sess.run(clip_disc_wei)
        # +--------------------+
        # Refine the generative network
        # Construct the noise input list of the generative network 构建生成网络的噪声输入列表
        #noise_inputs = get_noise_inputs()
        # +----------+
        # Construct the placeholder feed dictionary 构建占位符字典
        #ph_dict = dict(zip(noise_input_phs, noise_inputs))
        #ph_dict.update(dict(zip(gcn_fact_phs, gcn_facts)))
        #ph_dict.update({gnd_ph: gnd})
        #_, pre_g_loss, pre_g_output = sess.run([pre_gen_opt, pre_gen_loss, gen_output], feed_dict=ph_dict)

        if epoch%100==0:
            print('GAN-Train #%d, D-Loss: %f, G-Loss: %f'%(epoch, d_loss, g_loss))

    # +--------------------+
    # Conduct a prediction process
    # 执行一次预测操作
    # Construct the GCN factor list of the generative network 构建生成网络的GCN因子列表
    gcn_facts = []
    for k in range(t-window_size+1, t+2):
        # Read and normalize the adjacency matrix 读取并归一化邻接矩阵
        adj = read_data(name_pre, k, node_num, max_thres)/max_thres
        gcn_fact = get_gcn_fact(adj)
        gcn_facts.append(gcn_fact)
    # +----------+
    # Construct the noise input list of the generative network 构建生成网络的噪声输入列表
    noise_inputs = get_noise_inputs()
    # +----------+
    # Construct the placeholder feed dictionary 构建占位符字典
    ph_dict = dict(zip(noise_input_phs, noise_inputs))
    ph_dict.update(dict(zip(gcn_fact_phs, gcn_facts)))
    output = sess.run([gen_output], feed_dict=ph_dict)
    # +----------+
    # Reshape the prediction result 整理预测结果
    adj_est = np.reshape(output[0]*max_thres, (node_num, node_num))
    adj_est = (adj_est+adj_est.T)/2
    for r in range(node_num):
        adj_est[r, r] = 0
    for r in range(node_num):
        for c in range(node_num):
            if adj_est[r, c]<0.01:
                adj_est[r, c] = 0

    gnd = read_data(name_pre, t+2, node_num, max_thres)

    #print('adj_est')
    #for r in range(node_num):
    #    for c in range(node_num):
    #        print('%.2f'%(adj_est[c, r]), end=' ')
    #    print()
    #print('gnd')
    #for r in range(node_num):
    #    for c in range(node_num):
    #        print('%.2f'%(gnd[r, c]), end=' ')
    #    print()

    print('adj_est')
    for r in range(50):
        print('%.2f' % (adj_est[1, r]), end=' ')
    print()
    print('gnd')
    for r in range(50):
        print('%.2f' % (gnd[1, c]), end=' ')
    print()

    error = np.linalg.norm(gnd-adj_est, ord='fro')/(node_num*node_num)
    avg_error += error
    print('#%d Error: %f' % (t+2, error))

    edge_wei_KL = get_wei_KL(adj_est, gnd)
    avg_KL += edge_wei_KL
    print('#%d Edge Weight KL: %f' % (t + 2, edge_wei_KL))

    mis_rate = get_mis_rate(adj_est, gnd)
    avg_mis += mis_rate
    print('#%d Mismatch Rate: %f' % (t + 2, mis_rate))

    print()

    cal_count += 1

    f = open("+UCSB-LSTM_GAN_GCN-rror.txt", 'a+')
    #f = open("+KAIST-LSTM_GAN_GCN-error.txt", 'a+')
    f.write('%d %f' % (t + 2, error))
    f.write('\n')
    f.close()

    f = open("+UCSB-LSTM_GAN_GCN-KL.txt", 'a+')
    #f = open("+KAIST-LSTM_GAN_GCN_KL.txt", 'a+')
    f.write('%d %f' % (t + 2, edge_wei_KL))
    f.write('\n')
    f.close()

    f = open("+UCSB-LSTM_GAN_GCN-mis.txt", 'a+')
    #f = open("+KAIST-LSTM_GAN_GCN-mis.txt", 'a+')
    f.write('%d %f' % (t + 2, mis_rate))
    f.write('\n')
    f.close()

# +--------------------+
avg_error /= cal_count
avg_KL /= cal_count
avg_mis /= cal_count
# +--------------------+
f = open("+UCSB-LSTM_GAN_GCN-rror.txt", 'a+')
#f = open("+KAIST-LSTM_GAN_GCN-error.txt", 'a+')
f.write('Avg. Error %f' % (avg_error))
f.write('\n')
f.close()
# +--------------------+
f = open("+UCSB-LSTM_GAN_GCN-KL.txt", 'a+')
#f = open("+KAIST-LSTM_GAN_GCN_KL.txt", 'a+')
f.write('Avg. KL %f' % (avg_KL))
f.write('\n')
f.close()
# +--------------------+
f = open("+UCSB-LSTM_GAN_GCN-mis.txt", 'a+')
#f = open("+KAIST-LSTM_GAN_GCN-mis.txt", 'a+')
f.write('Avg. Mis %f' % (avg_mis))
f.write('\n')
f.close()
