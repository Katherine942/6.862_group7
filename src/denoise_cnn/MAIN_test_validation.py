import os.path
import math
import argparse
import time
import random
import bm3d 
import numpy as np
from collections import OrderedDict
import logging
import torch
from torch.utils.data import DataLoader


import utils_logger
import utils_image as util
import utils_option as option

from select_dataset import define_Dataset
from select_model import define_Model


'''
# --------------------------------------------
# training code for DnCNN
# --------------------------------------------
# 
# Reference:
@article{zhang2017beyond,
  title={Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={26},
  number={7},
  pages={3142--3155},
  year={2017},
  publisher={IEEE}
}
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def main(dataset_type = 'dncnn7', json_path='train_dncnn7.json',model_path = "results/dncnn_epoch1000_snrrand/models/latest_G.pth"):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''
    test_folder = 'test_results_4report'
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    opt = option.parse(parser.parse_args().opt, is_train=True)
    util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
#     init_iter, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    opt['path']['pretrained_netG'] = model_path
    opt['path']['root'] = test_folder
#     current_step = init_iter
    opt['datasets']['test']['dataset_type']=dataset_type
    
    border = 0
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    logger_name = 'test_bm3d_various_noises'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    logger.info('testing dataset type: {:s}'.format(opt['datasets']['test']['dataset_type']))
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            train_loader = DataLoader(train_set,
                                      batch_size=dataset_opt['dataloader_batch_size'],
                                      shuffle=dataset_opt['dataloader_shuffle'],
                                      num_workers=dataset_opt['dataloader_num_workers'],
                                      drop_last=True,
                                      pin_memory=True)
        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    if opt['merge_bn']:
        logger.info('^_^ -----merging bnorm----- ^_^')
        model.merge_bnorm_test()

    logger.info(model.info_network())
    model.init_train()
    logger.info(model.info_params())
    
    ####
    avg_psnr = 0.0
    avg_mse = 0.0
    avg_rmse = 0.0
    avg_psnr_bm3d = 0.0
    avg_mse_bm3d = 0.0
    avg_rmse_bm3d = 0.0
                
    idx = 0

    for test_data in test_loader:
        idx += 1
        image_name_ext = os.path.basename(test_data['L_path'][0])
        img_name, ext = os.path.splitext(image_name_ext)

#         img_dir = os.path.join(opt['path']['images'], img_name)
        
        

        model.feed_data(test_data)
        model.test()

        visuals = model.current_visuals()
        E_img = util.tensor2uint(visuals['E'])
        H_img = util.tensor2uint(visuals['H'])
        L_img = util.tensor2uint(visuals['L'])
        
        #------------------------
        # Denoise with bm3d to compare results
        #------------------------------
        psd = 255/(random.random()*10+10)
        bm3d_img = bm3d.bm3d(L_img, psd)

        # -----------------------
        # save estimated image E
        # -----------------------
       
        if idx%1 == 0:
            img_dir = os.path.join(opt['path']['images'],img_name)
            util.mkdir(img_dir)
            save_img_noisy_path = os.path.join(img_dir, '{:s}_{:s}_noisy.png'.format(img_name,dataset_type))
            save_img_path = os.path.join(img_dir, '{:s}_{:s}_test.png'.format(img_name,dataset_type))
            save_img_org_path = os.path.join(img_dir, '{:s}_org.png'.format(img_name))
            save_img_bm3d_path = os.path.join(img_dir, '{:s}_{:s}_bm3d.png'.format(img_name,dataset_type))
            util.imsave(L_img, save_img_noisy_path)
            util.imsave(E_img, save_img_path)
            util.imsave(H_img, save_img_org_path)
            util.imsave(bm3d_img, save_img_bm3d_path)

        # -----------------------
        # calculate PSNR
        # -----------------------
        current_psnr = util.calculate_psnr(E_img, H_img, border=border)

        logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))

        avg_psnr += current_psnr
                    
        # -----------------------
        # calculate MSE, RMSE
        # -----------------------
        current_mse = util.calculate_mse(E_img, H_img, border=border)

        logger.info('{:->4d}--> {:>10s} | {:<4.2f}MSE'.format(idx, image_name_ext, current_mse))
                    
        avg_mse += current_mse
                    
        current_rmse = util.calculate_rmse(E_img, H_img, border=border)

        logger.info('{:->4d}--> {:>10s} | {:<4.2f}RMSE'.format(idx, image_name_ext, current_rmse))
                    
        avg_rmse += current_rmse
        
                    
        
        # -----------------------
        # BM3D: calculate MSE, RMSE
        # -----------------------
        current_mse_bm3d = util.calculate_mse(bm3d_img, H_img, border=border)

        logger.info('{:->4d}--> {:>10s} | {:<4.2f}BM3D-MSE'.format(idx, image_name_ext, current_mse_bm3d))
                    
        avg_mse_bm3d += current_mse_bm3d
                    
        current_rmse_bm3d = util.calculate_rmse(bm3d_img, H_img, border=border)

        logger.info('{:->4d}--> {:>10s} | {:<4.2f}BM3D-RMSE'.format(idx, image_name_ext, current_rmse_bm3d))
                    
        avg_rmse_bm3d += current_rmse_bm3d
        # -----------------------
        # BM3D: calculate PSNR
        # -----------------------
        current_psnr_bm3d = util.calculate_psnr(bm3d_img, H_img, border=border)

        logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB - BM3D'.format(idx, image_name_ext, current_psnr_bm3d))

        avg_psnr_bm3d += current_psnr_bm3d

    avg_psnr = avg_psnr / idx
    avg_mse = avg_mse / idx
    avg_rmse = avg_rmse / idx
    logger.info('Average PSNR : {:<.2f}dB\n'.format(avg_psnr))
    logger.info('Average MSE : {:<.2f}\n'.format(avg_mse))
    logger.info('Average RMSE : {:<.2f}\n'.format(avg_rmse))        
    # BM3D:
    avg_psnr_bm3d = avg_psnr_bm3d / idx
    avg_mse_bm3d = avg_mse_bm3d / idx
    avg_rmse_bm3d = avg_rmse_bm3d / idx
    logger.info('Average PSNR - BM3D : {:<.2f}dB\n'.format(avg_psnr_bm3d))
    logger.info('Average MSE - BM3D : {:<.2f}\n'.format(avg_mse_bm3d))
    logger.info('Average RMSE - BM3D: {:<.2f}\n'.format(avg_rmse_bm3d))

    logger.info('End of testing')

if __name__ == '__main__':
#     #### First run: to study sensiivity of SNR:
#     snr_values = list(range(5,31,5))
#     print('running snr of: ', snr_values)
#     for snr in snr_values:
#         json_pth = 'json/train_dncnn7_snr'+str(snr)+'.json'
#         main(json_path=json_pth)

    ### Second run: run with best SNR and large number of epoch with checkpoints
    json_files = ['json/train_dncnn7_epoch1000_snrrand_rl.json','json/train_dncnn7_epoch1000_snrrand_gs.json']
    for dataset_type in ['dncnn7','dncnn7_rand','dncnn7_rand_rayleigh','dncnn7_gaussian']:
        json_pth = 'json/train_dncnn7_epoch1000_snrrand.json'
        main(dataset_type=dataset_type,json_path=json_pth)


