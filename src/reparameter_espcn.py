import torch
import torch.nn.functional as F

def transII_addbranch(kernels, biases):
    return sum(kernels), sum(biases)

def transIII_1x1_kxk(k1, b1, k2, b2,groups):
    if groups == 1:
        k = F.conv2d(k2, k1.permute(1, 0, 2, 3))      #
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

    print(list1)



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


model_path = "../model/model_best.pt"
model_outpath = "../model/model_rep.pt"

model = torch.load(model_path)
# print(model.keys())

conv1_0_list = []
conv1_1_list = []
conv1_2_list = []

conv2_0_list = []
conv2_1_list = []
conv2_2_list = []

conv3_0_list = []
conv3_1_list = []
conv3_2_list = []

conv4_0_list = []
conv4_1_list = []
conv4_2_list = []

for k in model.keys():
    if "conv1_0" in k:
        conv1_0_list.append(k)
    elif "conv1_1" in k:
        conv1_1_list.append(k)
    elif "conv1_2" in k:
        conv1_2_list.append(k)
    elif "conv2_0" in k:
        conv2_0_list.append(k)
    elif "conv2_1" in k:
        conv2_1_list.append(k)
    elif "conv2_2" in k:
        conv2_2_list.append(k)
    elif "conv3_0" in k:
        conv3_0_list.append(k)
    elif "conv3_1" in k:
        conv3_1_list.append(k)
    elif "conv3_2" in k:
        conv3_2_list.append(k)
    elif "conv4_0" in k:
        conv4_0_list.append(k)
    elif "conv4_1" in k:
        conv4_1_list.append(k)
    elif "conv4_2" in k:
        conv4_2_list.append(k)
    else:
        continue

conv1_0_list.sort()
conv1_1_list.sort()
conv1_2_list.sort()
conv2_0_list.sort()
conv2_1_list.sort()
conv2_2_list.sort()
conv3_0_list.sort()
conv3_1_list.sort()
conv3_2_list.sort()
conv4_0_list.sort()
conv4_1_list.sort()
conv4_2_list.sort()

# print(conv2_0_list)
# print(conv1_1_list)

#conv1 reparameter
model['conv1.weight'], model['conv1.bias'] = reparameter(model, conv1_0_list, conv1_1_list, conv1_2_list)

#conv2 reparameter
model['conv2.weight'], model['conv2.bias'] = reparameter(model, conv2_0_list, conv2_1_list, conv2_2_list)


# conv3 reparameter
model['conv3.weight'], model['conv3.bias'] = reparameter(model, conv3_0_list, conv3_1_list, conv3_2_list)

# conv4 reparameter
model['conv4.weight'], model['conv4.bias'] = reparameter(model, conv4_0_list, conv4_1_list, conv4_2_list)


torch.save(model,model_outpath)