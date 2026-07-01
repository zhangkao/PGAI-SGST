import torch, os, cv2, sys, math, shutil, copy, time, torchvision

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from model_pos_transformer_tp import *


from dataset import *
from utils_data import *
from loss_functions import *
from utils_score_torch import *
from utils_vis import *
from utils_data import normalize_data as norm_data

from patch_simple_torch2 import *




def get_guasspriors(b_s=2, shape_r=45, shape_c=80, channels = 8):
    priors_path = '/media/D/tsong/Project/uavsal/IIP_UAVSal_Saliency-main/'

    priormat_path = priors_path + 'gauss_priors_vr.mat'

    ims = h5io.loadmat(priormat_path)["PriorMaps"]
    if ims.shape[0] != shape_r or ims.shape[1] != shape_c:
        ims_rs = np.zeros((shape_r, shape_c, ims.shape[2]), np.uint8)
        for i in range(ims.shape[2]):
            ims_rs[:, :, i] = padding(ims[:, :, i], shape_r, shape_c, 1)
        ims = ims_rs

    ims = np.expand_dims(ims, axis=0)
    ims = np.repeat(ims, b_s, axis=0)

    return ims



def train(method_name='vr',
          iosize=[360, 720, 45, 90],
          time_dims=20,
          num_stblock=2,
          bias_type=[1, 0, 0],
          batch_size=20,
          epochs=20,
          pre_model_path=''):

    tmdir = saveModelDir + method_name
    save_model_path = tmdir + '/' + method_name + '_'
    if not os.path.exists(tmdir):
        os.makedirs(tmdir)

    #################################################################
    # Build the model
    #################################################################
    print("Build Model: " + method_name)
    patches = SpherePatcher.generate_patches(delta_theta=20, min_phi_div=4)
    original_vit = create_model('vit_base_patch16_224', pretrained=False)
    model = RobustDynamicViT(original_vit, patches).to(device)
    # original_vit = create_model('vit_base_patch16_224', pretrained=True)
    # model = RobustDynamicViT(original_vit, learnable_interp=True).to(device)


    shape_r, shape_c, shape_r_out, shape_c_out = iosize
    criterion = loss_fu

    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad == True], lr=1e-5,
                                 betas=(0.9, 0.999), weight_decay=0.0005)

    print("Training Model")
    min_val_loss = 10000
    num_patience = 0
    if IS_EARLY_STOP:
        max_patience = Max_patience
    else:
        max_patience = epochs + 1


    for epoch in range(epochs):
        print("\nEpochs: %d / %d " % (epoch + 1, epochs))
        for phase in ['train', 'val']:
            num_step = 0
            run_loss = 0.0
            if phase == 'train':
                model.train()
                shuffle = Shuffle_Train
                Max_TrainValFrame = Max_TrainFrame
            else:
                model.eval()
                shuffle = False
                Max_TrainValFrame = Max_ValFrame

            # patch_list, videos_list, vidmaps_list, vidfixs_list = read_video_list_new(train_patch_path, train_dataDir, phase, shuffle=shuffle, ext='.mp4')
            patch_list, videos_list, vidmaps_list, vidfixs_list = read_video_list(train_patch_path, train_dataDir,
                                                                                      phase, shuffle=shuffle,
                                                                                      ext='.mp4')

            x_cb_gauss = get_guasspriors(batch_size*time_dims, shape_r, shape_c, channels=8).transpose((0, 3, 1, 2))
            x_cb_gauss = torch.tensor(x_cb_gauss).float().to(device)

            for idx_video in range(len(videos_list)):
                print("Videos: %d / %d, %s with data from: %s" % (
                    idx_video + 1, len(videos_list), phase.upper(), videos_list[idx_video]))

                vidmaps = preprocess_vidmaps(vidmaps_list[idx_video], shape_r_out, shape_c_out, Max_TrainValFrame)
                vidfixs = preprocess_vidfixs(vidfixs_list[idx_video], shape_r_out, shape_c_out, Max_TrainValFrame)
                vidimgs, nframes, height, width = preprocess_videos(videos_list[idx_video], shape_r, shape_c,
                                                                    Max_TrainValFrame, mode='RGB', normalize=False)

                nframes = min(min(vidfixs.shape[0], vidmaps.shape[0]), nframes)

                count_bs = nframes // time_dims
                trainFrames = count_bs * time_dims

                vidimgs = vidimgs[0:trainFrames].transpose((0, 3, 1, 2))
                vidgaze = np.concatenate((vidmaps[0:trainFrames], vidfixs[0:trainFrames]), axis=-1).transpose(
                    (0, 3, 1, 2))

                count_input = batch_size * time_dims
                bs_steps = math.ceil(count_bs / batch_size)
                video_loss = 0.0
                # x_state = None
                for idx_bs in range(bs_steps):
                    x_imgs = vidimgs[idx_bs * count_input:(idx_bs + 1) * count_input]

                    patch_batch, meta_list = erp_to_patches(torch.tensor(norm_data(x_imgs)).float().to(device), patches)

                    patches_tensor = patch_batch.contiguous().requires_grad_(True)
                    y_gaze = vidgaze[idx_bs * count_input:(idx_bs + 1) * count_input]
                    # x_patch_name = patches_name[idx_bs * count_input:(idx_bs + 1) * count_input]

                    if not np.any(y_gaze, axis=(2, 3)).all():
                        continue


                    # x_patch = norm_data(patches_tensor)
                    x_patch = patches_tensor
                    # print(x_patch.shape)


                    y_gaze = torch.tensor(y_gaze).float()

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):

                        x_imgs = torch.tensor(norm_data(x_imgs)).float()
                        outputs = model(x_patch.to(device), meta_list, shape_r_out, shape_c_out)

                        with torch.set_grad_enabled(True):

                            loss = criterion(outputs, y_gaze.to(device))
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                    batch_loss = loss.data.item()
                    video_loss += batch_loss
                    run_loss += batch_loss
                    num_step += 1

                    print("    Batch: [%d / %d], %s loss : %.4f " % (idx_bs + 1, bs_steps, phase.upper(), batch_loss))

                print("    Mean %s loss: %.4f " % (phase.upper(), video_loss / bs_steps))

            mean_run_loss = run_loss / num_step
            print("Epoch: %d / %d, Mean %s loss: %.4f" % (epoch + 1, epochs, phase.upper(), mean_run_loss))

        if not IS_BEST_ONLY:
            output_modename = save_model_path + "%02d_%.4f.pth" % (epoch, mean_run_loss)
            torch.save(model, output_modename)
        if mean_run_loss < min_val_loss:
            min_val_loss = mean_run_loss


            num_patience = 0
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            num_patience += 1
            if num_patience >= max_patience:
                print('Early stop')
                break

    # Save the best model
    finalmode_name = save_model_path + "final.pth"
    model.load_state_dict(best_model_wts)
    torch.save(model, finalmode_name)


def test(input_path, output_path, method_name,
         saveFrames=float('inf'),
         time_dims=20,
         iosize=[360, 720, 45, 90],
         batch_size=4):


    model_path = "/media/D/tsong/Project/vr_transformer/weights19/model_pos_avs_final/vr/vr_final.pth"

    model = torch.load(model_path)
    model = model.to(device)

    output_path = output_path + method_name + '/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    shape_r, shape_c, shape_r_out, shape_c_out = iosize

    file_names = [f for f in os.listdir(input_path) if (f.endswith('.avi') or f.endswith('.AVI') or f.endswith('.mp4'))]
    file_names.sort()
    nb_videos_test = len(file_names)

    x_cb_gauss = get_guasspriors(batch_size * time_dims, shape_r, shape_c, channels=8).transpose((0, 3, 1, 2))
    x_cb_gauss = torch.tensor(x_cb_gauss).float().to(device)

    model.eval()


    with torch.no_grad():
        for idx_video in range(nb_videos_test):
            print("%d/%d   " % (idx_video + 1, nb_videos_test) + file_names[idx_video])

            ovideo_path = output_path + (file_names[idx_video])[:-4] + '.mat'
            if os.path.exists(ovideo_path):
                continue

            ivideo_path = input_path + file_names[idx_video]
            vidimgs, nframes, height, width = preprocess_videos(ivideo_path, shape_r, shape_c, saveFrames, mode='RGB',
                                                                normalize=False)

            count_bs = nframes // time_dims
            isaveframes = count_bs * time_dims
            vidimgs = vidimgs[0:isaveframes].transpose((0, 3, 1, 2))

            pred_mat = np.zeros((isaveframes, height, width, 1), dtype=np.uint8)
            count_input = batch_size * time_dims
            bs_steps = math.ceil(count_bs / batch_size)
            x_state = None
            for idx_bs in range(bs_steps):
                x_imgs = vidimgs[idx_bs * count_input:(idx_bs + 1) * count_input]

                patches = SpherePatcher.generate_patches(delta_theta=20, min_phi_div=4)
                patch_batch, meta_list = erp_to_patches(torch.tensor(norm_data(x_imgs)).float().to(device), patches)
                patches_tensor = patch_batch.contiguous()

                # x_patch = norm_data(patches_tensor)
                x_patch = patches_tensor

                x_imgs = torch.tensor(norm_data(x_imgs)).float()

                bs_out = model(x_patch.to(device), meta_list, shape_r_out, shape_c_out, time_dims, x_cb_gauss)

                bs_out = bs_out.data.cpu().numpy()

                for idx_pre in range(bs_out.shape[0]):
                    isalmap = postprocess_predictions(bs_out[idx_pre, 0, :, :], height, width)
                    pred_mat[idx_bs * count_input + idx_pre, :, :, 0] = np2mat(isalmap)

            iSaveFrame = min(isaveframes, saveFrames)
            pred_mat = pred_mat[0:iSaveFrame, :, :, :].transpose((1, 2, 3, 0))

            h5io.savemat(ovideo_path, {'salmap': pred_mat})


IS_EARLY_STOP = True
IS_BEST_ONLY = False
Shuffle_Train = True
Max_patience = 10
Max_TrainFrame = float('inf')
Max_ValFrame = float('inf')
saveFrames = float('inf')
ext='.mp4'

train_patch_path = '/media/D/mayun/SVGC_AVA/training/'
saveModelDir = "./weights/model_pos_svgc/"
train_dataDir = '/media/D/mayun/SVGC_AVA/train_videos/'


if __name__ == '__main__':

    method_name = 'vr'
    epochs = 40
    batch_size = 5 #5

    # time_dims = 5
    time_dims = 20
    iosize = [360, 720, 360, 720]
    train(batch_size = batch_size, iosize=iosize, time_dims = time_dims)



    # test_input_path = '/media/D/mayun/SVGC_AVA/videos/videos/'
    # test_result_path = '/media/D/tsong/SVGC_AVA/avs_svgc/'

    
    # test_output_path = test_result_path + 'Saliency/'
    # test(test_input_path, test_output_path, method_name=method_name, saveFrames=saveFrames, iosize=iosize,
    #      batch_size=batch_size, time_dims=time_dims)

    # test_dataDir = '/media/D/mayun/SVGC_AVA/videos/'

    # DataSet_Test = 'vr'
    # evalscores_vid_torch(test_dataDir, test_result_path, DataSet=DataSet_Test, MethodNames=[method_name], batch_size=32)
