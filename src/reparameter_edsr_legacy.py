import torch
import torch.nn.functional as F
import argparse

def transII_addbranch(kernels, biases):
    return sum(kernels), sum(biases)

def transIII_1x1_kxk(k1, b1, k2, b2,groups):
    if groups == 1:
        k = F.conv2d(k2, k1.permute(1, 0, 2, 3))     
        b_hat = (k2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3))
    elif groups == 10:
        k = k1 * k2
        b_hat = (k2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3))
    else:
        k_slices = []
        b_slices = []
        k1_T = k1.permute(1, 0, 2, 3)
        k1_group_width = k1.size(0) // groups
        k2_group_width = k2.size(0) // groups
        for g in range(groups):
            k1_T_slice = k1_T[:, g*k1_group_width:(g+1)*k1_group_width, :, :]
            k2_slice = k2[g*k2_group_width:(g+1)*k2_group_width, :, :, :]
            k_slices.append(F.conv2d(k2_slice, k1_T_slice))
            b_slices.append((k2_slice * b1[g*k1_group_width:(g+1)*k1_group_width].reshape(1, -1, 1, 1)).sum((1, 2, 3)))
        k, b_hat = transIV_depthconcat(k_slices, b_slices)
    return k, b_hat + b2

def transIV_depthconcat(kernels, biases):
    return torch.cat(kernels, dim=0), torch.cat(biases)

def transVI_multiscale(kernel, target_kernel_size):
    H_pixels_to_pad = (target_kernel_size - kernel.size(2)) // 2
    W_pixels_to_pad = (target_kernel_size - kernel.size(3)) // 2
    return F.pad(kernel, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad])


#reparameter
def reparameter(model,list1, list2, list3):

    # print(list1)



    k_0_01, b_0_01 = transIII_1x1_kxk(model[list1[3]],model[list1[2]],model[list1[5]],model[list1[4]],groups=1)
    k_0_012, b_0_012 = transIII_1x1_kxk(model[list1[1]],model[list1[0]],k_0_01,b_0_01,groups=1)

    k_1_01, b_1_01 = transIII_1x1_kxk(model[list2[1]],model[list2[0]],model[list2[3]],model[list2[2]],groups=1)

    k_012, b_012 = transII_addbranch((k_0_012,k_1_01,model[list3[1]]),(b_0_012,b_1_01,model[list3[0]]))

    conv2_list = list1 + list2 + list3

    for i in conv2_list:
        # print(i)
        del model[i]

    # print(model.keys())

    return k_012, b_012

parser = argparse.ArgumentParser(description='PyTorch EDSR')
parser.add_argument('--model_folder', type=str, default='experiment/model', help='model folder to use')

model_folder = parser.parse_args().model_folder
model_path = model_folder + '/model_best.pt'
model_outpath = model_folder + '/model_rep.pt'

model = torch.load(model_path)
# print(model.keys())

res0_conv1_0_list = []
res0_conv1_1_list = []
res0_conv1_2_list = []

res0_conv2_0_list = []
res0_conv2_1_list = []
res0_conv2_2_list = []

res1_conv1_0_list = []
res1_conv1_1_list = []
res1_conv1_2_list = []

res1_conv2_0_list = []
res1_conv2_1_list = []
res1_conv2_2_list = []


conv_0_list = []
conv_1_list = []
conv_2_list = []


for k in model.keys():
    if "body.0.body.0.conv_0" in k:
        res0_conv1_0_list.append(k)
    elif "body.0.body.0.conv_1" in k:
        res0_conv1_1_list.append(k)
    elif "body.0.body.0.conv_2" in k:
        res0_conv1_2_list.append(k)
    elif "body.0.body.2.conv_0" in k:
        res0_conv2_0_list.append(k)
    elif "body.0.body.2.conv_1" in k:
        res0_conv2_1_list.append(k)
    elif "body.0.body.2.conv_2" in k:
        res0_conv2_2_list.append(k)

    elif "body.1.body.0.conv_0" in k:
        res1_conv1_0_list.append(k)
    elif "body.1.body.0.conv_1" in k:
        res1_conv1_1_list.append(k)
    elif "body.1.body.0.conv_2" in k:
        res1_conv1_2_list.append(k)
    elif "body.1.body.2.conv_0" in k:
        res1_conv2_0_list.append(k)
    elif "body.1.body.2.conv_1" in k:
        res1_conv2_1_list.append(k)
    elif "body.1.body.2.conv_2" in k:
        res1_conv2_2_list.append(k)
    elif "body.2.conv_0" in k:
        conv_0_list.append(k)
    elif "body.2.conv_1" in k:
        conv_1_list.append(k)
    elif "body.2.conv_2" in k:
        conv_2_list.append(k)
   
    else:
        continue


res0_conv1_0_list.sort()
res0_conv1_1_list.sort()
res0_conv1_2_list.sort()
# print(res0_conv1_0_list)

res0_conv2_0_list.sort()
res0_conv2_1_list.sort()
res0_conv2_2_list.sort()


res1_conv1_0_list.sort()
res1_conv1_1_list.sort()
res1_conv1_2_list.sort()

res1_conv2_0_list.sort()
res1_conv2_1_list.sort()
res1_conv2_2_list.sort()

conv_0_list.sort()
conv_1_list.sort()
conv_2_list.sort()


model['body.0.body.0.conv.weight'], model['body.0.body.0.conv.bias'] = reparameter(model, res0_conv1_0_list, res0_conv1_1_list, res0_conv1_2_list)
model['body.0.body.2.conv.weight'], model['body.0.body.2.conv.bias'] = reparameter(model, res0_conv2_0_list, res0_conv2_1_list, res0_conv2_2_list)

model['body.1.body.0.conv.weight'], model['body.1.body.0.conv.bias'] = reparameter(model, res1_conv1_0_list, res1_conv1_1_list, res1_conv1_2_list)
model['body.1.body.2.conv.weight'], model['body.1.body.2.conv.bias'] = reparameter(model, res1_conv2_0_list, res1_conv2_1_list, res1_conv2_2_list)

print(len(res0_conv1_0_list), len(res0_conv1_1_list), len(res0_conv1_2_list))
print(res0_conv1_0_list, res0_conv1_1_list, res0_conv1_2_list)

model['body.2.conv.weight'], model['body.2.conv.bias'] = reparameter(model, conv_0_list, conv_1_list, conv_2_list)

# print(model.keys())


torch.save(model,model_outpath)