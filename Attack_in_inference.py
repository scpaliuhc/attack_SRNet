from copyreg import pickle
from re import T
import tensorflow as tf
# from model_loss_cfg_dat import *
import numpy as np
import os
import argparse
from tqdm import tqdm
from util import print_log,PrintColor
import cv2
from attacks import *

############################################
# i_s: 源风格图
# i_t: 标准的目标文字
# mast_t: i_s中的文字mask
# t_b: i_s中背景
# t_f: 目标图片
# t_sk: i_t中文字的骨架（已经有i_s中文字的风格）
# t_t: 具有i_s文字风格的目标文字（没有背景）
############################################

def pars():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--gpu', help = 'gpu id', default = 1)
    parser.add_argument('--input_dir', help = 'Directory containing xxx_i_s and xxx_i_t with same prefix')
    parser.add_argument('--save_dir', default=None, help = 'Directory to save result')
    parser.add_argument('--checkpoint', help = 'tensorflow ckpt')
    parser.add_argument('--AE_m', default='BIM',help = 'attack')
    parser.add_argument('--editor', default='SRNet', help = 'editor model')
    # parser.add_argument('--eps', default=0.02, help = 'epsilon', type=float)
    parser.add_argument('--Linf', default=0.3, help = 'epsilon', type=float)
    parser.add_argument('--iter', default=20, help = 'iterations', type=int)
    parser.add_argument('--u', default=0.8, help = 'decay factor in MI-FGSM', type=float)
    args = parser.parse_args()
    return args

def SRNet_init(args):
    from model_loss_cfg_data.SRNet_model import SRNet
    import model_loss_cfg_data.SRNet_cfg as cfg

    print_log('model compiling start.', content_color = PrintColor['yellow'])
    model = SRNet(shape = cfg.data_shape, name = 'predict')
    print_log('model compiled.', content_color = PrintColor['yellow'])

    sess=tf.InteractiveSession(graph=model.graph)
    saver = tf.train.Saver(tf.global_variables())
    print_log('weight loading start.', content_color = PrintColor['yellow'])
    try:
        saver.restore(sess, args.checkpoint)
    except:
        saver.restore(sess, cfg.predict_ckpt_path)
    print_log('weight loaded.', content_color = PrintColor['yellow'])
    
    print_log('record all interfaces.', content_color = PrintColor['yellow'])
    loss_type=['t_sk','t_t','t_b','t_f','all']
    loss_dic={}
    model_input_dic={}
    model_label_dic={}
    model_out_dic={}
    adv_dic={}

    model_input_dic['i_s']=model.i_s
    model_input_dic['i_t']=model.i_t

    model_out_dic['sk']=model.o_sk
    model_out_dic['t']=model.o_t
    model_out_dic['b']=model.o_b
    model_out_dic['f']=model.o_f

    model_label_dic['sk']=model.t_sk
    model_label_dic['t']=model.t_t
    model_label_dic['b']=model.t_b
    model_label_dic['f']=model.t_f
    model_label_dic['mask']=model.mask_t
    
    #这里的损失值越小，说明模型性能越好，攻击的目的是增大损失
    loss_dic['sk']=model.g_loss_detail[0]
    loss_dic['t']=model.g_loss_detail[1]
    loss_dic['b']= -1*tf.reduce_mean(tf.abs(model_input_dic['i_s'] - model_out_dic['b']))
    # model.g_loss_detail[3]
    loss_dic['f']=model.g_loss_detail[5]
    loss_dic['all']= cfg.lbd_sk*loss_dic['sk']+cfg.lbd_t*loss_dic['t']+cfg.lbd_b*loss_dic['b']+cfg.lbd_f*loss_dic['f']

    print_log('completed record.', content_color = PrintColor['yellow'])
    return model,sess,loss_dic,model_input_dic,model_label_dic,model_out_dic

def SRNet_attack(args):
    from model_loss_cfg_data.SRNet_datagen import get_input_data_with_labels, get_input_data
    import model_loss_cfg_data.SRNet_cfg as cfg
    
    model,sess,loss_dic,model_input_dic,model_label_dic,model_out_dic=SRNet_init(args)
    
    key_loss=[key for key in loss_dic.keys()]
    # key_input=[key for key in model_input_dic.keys()]
    # key_label_out=[key for key in model_label_dic.keys()]
    print_log(f'keys of loss:{key_loss}', content_color = PrintColor['yellow'])
    
    print_log(f'define node for {args.AE_m}.', content_color = PrintColor['yellow'])
    adv_dic={}
    if args.AE_m=='FGSM':
        for key in key_loss:
            adv_dic[key]=FGSM_tf_pre(loss_dic[key],model_input_dic['i_s'],args.Linf,[-1.,1.])
    elif args.AE_m=='BIM' or args.AE_m=='PGD':
        alpha=args.eps/args.iter
        for key in key_loss:
            adv_dic[key]=FGSM_tf_pre(loss_dic[key],model_input_dic['i_s'],alpha)
    elif args.AE_m=='MI-FGSM':
        alpha=args.Linf/args.iter
        grad_dic={}
        grad_l1_norm_dic={}
        g_t=tf.placeholder(dtype = tf.float32, shape = [None, 64, None, 3])
        for key in key_loss:
            adv_dic[key],grad_dic[key],grad_l1_norm_dic[key]=MI_FGSM_tf_pre(loss_dic[key],model_input_dic['i_s'], g_t, alpha, args.u, [-1., 1.])

    
    #load data
    print_log('prepare data.', content_color = PrintColor['yellow'])
    try:
        input_data_list=get_input_data_with_labels(args.input_dir)# normalize, resize
    except:
        input_data_list=get_input_data_with_labels(cfg.predict_data_dir)
    assert len(input_data_list)>0

    save_dir=os.path.join('datasets/results',args.AE_m) if args.save_dir is None else args.save_dir
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print_log(f'start attack.', content_color = PrintColor['yellow'])
    loss_pic={}
    result={}#存储所有data对应的output以及对抗样本
    for data in tqdm(input_data_list):
        i_t, i_s, t_sk, t_t, t_b, t_f, mask_t, ori_shape, data_name = data
        
        print_log(f'Attacking {data_name}.', content_color = PrintColor['yellow'])

        print_log('\t producing fake GT.', content_color = PrintColor['blue'])
        fgt_np = sess.run(model_out_dic,feed_dict={model_input_dic['i_s']:i_s,
        model_input_dic['i_t']:i_t})

        print_log(f'\t producing AE.', content_color = PrintColor['blue'])
        if args.AE_m=='FGSM':
            adv_gt=sess.run(adv_dic,feed_dict={
                model_input_dic['i_s']:i_s,
                model_input_dic['i_t']:i_t,
                model_label_dic['sk']:t_sk,
                model_label_dic['t']:t_t,
                model_label_dic['b']:t_b,
                model_label_dic['f']:t_f,
                model_label_dic['mask']:mask_t,
            })
        elif args.AE_m=='BIM' or args.AE_m=='PGD':
            adv_gt={}
            loss_temp={}
            for key in key_loss:#针对不同阶段发起攻击
                adv_gt[key]=i_s
                loss_temp[key]=[]
                if args.AE_m=='PGD':
                    adv_gt[key]+=np.random.uniform(-args.Linf, args.Linf, i_s.shape)
                    adv_gt[key]=np.clip(adv_gt[key], -1., 1.)
                
                for i in range(args.iter):
                    adv_gt[key],loss=sess.run([adv_dic[key],loss_dic[key]],feed_dict={
                                    model_input_dic['i_s']:adv_gt[key],
                                    model_input_dic['i_t']:i_t,
                                    model_label_dic['sk']:t_sk,
                                    model_label_dic['t']:t_t,
                                    model_label_dic['b']:t_b,
                                    model_label_dic['f']:t_f,
                                    model_label_dic['mask']:mask_t
                                    })
                    loss_temp[key].append(loss)
                    adv_gt[key]=np.clip(adv_gt[key], i_s-args.Linf, i_s+args.Linf)
                    adv_gt[key]=np.clip(adv_gt[key], -1., 1.)
            loss_pic[data_name]=loss_temp
        elif args.AE_m=='MI-FGSM':
            adv_gt={}
            for key in key_loss:
                adv_gt[key]=i_s
                g=np.zeros(i_s.shape)
                for i in range(args.iter):
                    grad_, grad_l1_norm_=sess.run([ grad_dic[key], grad_l1_norm_dic[key]], feed_dict={
                                    model_input_dic['i_s']:adv_gt[key],
                                    model_input_dic['i_t']:i_t,
                                    model_label_dic['sk']:t_sk,
                                    model_label_dic['t']:t_t,
                                    model_label_dic['b']:t_b,
                                    model_label_dic['f']:t_f,
                                    model_label_dic['mask']:mask_t,
                                    g_t:g
                                    })
                    g=args.u* g+(grad_/grad_l1_norm_)
                    adv_gt[key]=adv_gt[key]+alpha*np.sign(g)
                    adv_gt[key]=np.clip(adv_gt[key],-1.,1.)

    
        print_log(f'\t predicting on AE.', content_color = PrintColor['blue'])
        adv_gt_con=np.concatenate([adv_gt[key] for key in key_loss],axis=0)
        i_t_=np.tile(i_t,(adv_gt_con.shape[0],1,1,1))
        o_f_,o_b_,o_t_,o_sk_=sess.run([model.o_f,model.o_b,model.o_t,model.o_sk],feed_dict = {model_input_dic['i_s']:adv_gt_con,
                                             model_input_dic['i_t']:i_t_,})

        print_log(f'\t saving', content_color = PrintColor['blue'])

        o_wo_p = cv2.resize(((fgt_np['f'][0] + 1.) * 127.5).astype(np.uint8), ori_shape)
        cv2.imwrite(os.path.join(save_dir, data_name + f'_f_out_ori.png'), o_wo_p, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        for id, key in enumerate(key_loss):
            # if key != 'all':
            #     if key=='sk':
            #         o_wo_p=cv2.resize((fgt_np[key][0] * 255.).astype(np.uint8), ori_shape, interpolation=cv2.INTER_NEAREST)
            #     else:
            #         o_wo_p = cv2.resize(((fgt_np[key][0] + 1.) * 127.5).astype(np.uint8), ori_shape)
            
            
            adv_img=cv2.resize(((adv_gt[key][0] + 1.) * 127.5).astype(np.uint8), ori_shape)
            f=cv2.resize(((o_f_[id] + 1.) * 127.5).astype(np.uint8), ori_shape)
            b=cv2.resize(((o_b_[id] + 1.) * 127.5).astype(np.uint8), ori_shape)
            t=cv2.resize(((o_t_[id] + 1.) * 127.5).astype(np.uint8), ori_shape)
            sk=cv2.resize(((o_sk_[id] + 1.) * 127.5).astype(np.uint8), ori_shape)
            cv2.imwrite(os.path.join(save_dir, f'{data_name}_{key}_inp.png'),adv_img)
            cv2.imwrite(os.path.join(save_dir, f'{data_name}_{key}_out_f.png'),f)
            cv2.imwrite(os.path.join(save_dir, f'{data_name}_{key}_out_b.png'),b)
            cv2.imwrite(os.path.join(save_dir, f'{data_name}_{key}_out_t.png'),t)
            cv2.imwrite(os.path.join(save_dir, f'{data_name}_{key}_out_sk.png'),sk)
    print_log(f'attack finished.', content_color = PrintColor['yellow'])

    with open(os.path.join(save_dir,'record.b'),'wb') as f:
        pickle.dump([loss_pic,result],f)


def main():
    args=pars()
    # gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)


    #preprocess
    
    
    if args.editor=='SRNet':
        SRNet_attack(args) 
            




if __name__=='__main__':
    main()