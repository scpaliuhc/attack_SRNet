import tensorflow as tf
# from model_loss_cfg_dat import *
import numpy as np
import os
import argparse
from tqdm import tqdm
from util import print_log,PrintColor
import cv2
from attacks import *

def pars():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--gpu', help = 'gpu id', default = 1)
    parser.add_argument('--input_dir', help = 'Directory containing xxx_i_s and xxx_i_t with same prefix')
    parser.add_argument('--save_dir', default=None, help = 'Directory to save result')
    parser.add_argument('--checkpoint', help = 'tensorflow ckpt')
    parser.add_argument('--c', default=1, help = 'balance', type=float)
    parser.add_argument('--iter', default=2000, help = 'iterations', type=int)
    parser.add_argument('--lr', default=1000, help = 'learning rate', type=float) #1000
    args = parser.parse_args()
    return args

def main(args):
    from model_loss_cfg_data.SRNet_datagen import get_input_data_with_labels, get_input_data
    from model_loss_cfg_data.SRNet_model import SRNet
    import model_loss_cfg_data.SRNet_cfg as cfg

    print_log('model compiling start.', content_color = PrintColor['yellow'])
    model = SRNet(shape = cfg.data_shape, name = 'predict', cw_w=True)
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
    loss_dic={}
    model_input_dic={}
    model_label_dic={}
    model_out_dic={}
    adv_dic={}
    
    loss_dic['sk']=model.g_loss_detail[0]
    loss_dic['t']=model.g_loss_detail[1]
    loss_dic['b']=model.g_loss_detail[3]
    loss_dic['f']=model.g_loss_detail[5]
    loss_dic['all']= loss_dic['sk']+loss_dic['t']+loss_dic['b']+loss_dic['f']
    
    model_input_dic['i_s']=model.i_s
    model_input_dic['i_t']=model.i_t
    model_input_dic['w']=model.w
    
    model_label_dic['sk']=model.t_sk
    model_label_dic['t']=model.t_t
    model_label_dic['b']=model.t_b
    model_label_dic['f']=model.t_f
    model_label_dic['mask']=model.mask_t
    
    model_out_dic['sk']=model.o_sk
    model_out_dic['t']=model.o_t
    model_out_dic['b']=model.o_b
    model_out_dic['f']=model.o_f
    print_log('completed record.', content_color = PrintColor['yellow'])
    
    adv_dic={}
    grad={}
    loss_cw={}
    ori=tf.placeholder(dtype = tf.float32, shape = [None, 64, None, 3])
    key_loss=[key for key in loss_dic.keys()]
    c_dic={'sk':1.,'t':10.,'b':0.1,'f':0.1,'all':0.01}
    for key in key_loss:
        loss1=loss_dic[key]#越大越好
        loss2=tf.reduce_mean(tf.square(tf.tanh(model_input_dic['w'])-ori))
        loss=loss2-c_dic[key]*loss1
        loss_cw[key]={'all':loss, 'loss1':loss1, 'loss2':loss2}
        grad[key]=tf.gradients(loss,model_input_dic['w'])[0]

    try:
        input_data_list=get_input_data_with_labels(args.input_dir)# normalize, resize
    except:
        input_data_list=get_input_data_with_labels(cfg.predict_data_dir)
    assert len(input_data_list)>0
    save_dir=os.path.join('datasets/results','CW') if args.save_dir is None else args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print_log(f'start attack.', content_color = PrintColor['yellow'])
    for data in tqdm(input_data_list):
        i_t, i_s, t_sk, t_t, t_b, t_f, mask_t, ori_shape, data_name = data
        print_log(f'Attacking {data_name}.', content_color = PrintColor['yellow']) 
        adv_gt={}
        adv_w={}
        for key in key_loss:
            #init 
            w=np.zeros(i_s.shape,dtype=np.float)
            for i in range(args.iter):
                g,l_all,l_1,l_2=sess.run([grad[key],loss_cw[key]['all'],loss_cw[key]['loss1'],loss_cw[key]['loss2']], feed_dict={
                    model_input_dic['w']:w,
                    model_input_dic['i_t']:i_t,
                    model_label_dic['sk']:t_sk,
                    model_label_dic['t']:t_t,
                    model_label_dic['b']:t_b,
                    model_label_dic['f']:t_f,
                    model_label_dic['mask']:mask_t,
                    ori:i_s,
                })
                w=w-args.lr*g
                print_log(f'\t\t CW {data_name} {key} {i}/{args.iter} {l_all:.6f} {l_1:.6f} {l_2:.6f}.', content_color = PrintColor['green'])


            adv_gt[key]=np.tanh(w)
            adv_w[key]=w
            o_f_gt=sess.run(model.o_f,feed_dict = {model_input_dic['w']     :w,model_input_dic['i_t']:i_t,})
            adv_img=cv2.resize(((adv_gt[key][0] + 1.) * 127.5).astype(np.uint8), ori_shape)
            pred_img=cv2.resize(((o_f_gt[0] + 1.) * 127.5).astype(np.uint8), ori_shape)
            cv2.imwrite(os.path.join(save_dir, f'{data_name}_{key}_adv.png'),adv_img)
            cv2.imwrite(os.path.join(save_dir, f'{data_name}_{key}_out.png'),pred_img)
        
        # print_log(f'\t predicting on AE.', content_color = PrintColor['blue'])
        # adv_w_con=np.concatenate([adv_w[key] for key in key_loss],axis=0)
        # i_t_=np.tile(i_t,(adv_w_con.shape[0],1,1,1))
        # o_f_gt=sess.run(model.o_f,feed_dict = {model_input_dic['w']:adv_w_con,
        #                                      model_input_dic['i_t']:i_t_,})

        # print_log(f'\t saving', content_color = PrintColor['blue'])
        # for id, key in enumerate(key_loss):
        #     adv_img=cv2.resize(((adv_gt[key][0] + 1.) * 127.5).astype(np.uint8), ori_shape)
        #     pred_img=cv2.resize(((o_f_gt[id] + 1.) * 127.5).astype(np.uint8), ori_shape)
        #     cv2.imwrite(os.path.join(save_dir, f'{data_name}_{key}_adv.png'),adv_img)
        #     cv2.imwrite(os.path.join(save_dir, f'{data_name}_{key}_out.png'),pred_img)
        
    print_log(f'attack finished.', content_color = PrintColor['yellow'])

if __name__=='__main__':
    main(pars())