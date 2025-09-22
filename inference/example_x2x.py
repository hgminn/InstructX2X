import torch, torchvision
from rse_x2x import LocalEditor

#------------------------------------------[[[-------------#
# Path to target image
input_image = "imgs/example.jpg"  
# Text used to guide image translation
edit_text = "Add the finding of pleural effusion"
# Parameters
rse_tau      = 0.1 # Mask threshold hyper-parameter
img_size     = 512 # Input image size
s_img, s_txt = 1.5, 7.5 # Guidance strengths for IP2P
#-------------------------------------------------------#

rse_editor_model = LocalEditor()

# Read and preprocess image
image_tensor0 = torchvision.io.read_image(input_image).float() / 255 * 2.0 - 1
if image_tensor0.shape[0] == 1:
    image_tensor0 = image_tensor0.repeat(3, 1, 1) 

image_tensor  = torchvision.transforms.Resize(size = img_size)(image_tensor0)
print('Loading and processing', input_image, f'({image_tensor0.shape} -> {image_tensor.shape})')

print('Editing image (rse)')
edited_image, heatmap = rse_editor_model(image_tensor, 
                                         edit = edit_text, 
                                         check_size = False,
                                         scale_txt = s_txt,
                                         scale_img = s_img,
                                         mask_threshold = rse_tau,
                                         return_heatmap = True)

print('Editing image(without rse)')
edited_image_ip2p, _ = rse_editor_model(image_tensor, 
                                        edit = edit_text, 
                                        check_size = False,
                                        scale_txt = s_txt,
                                        scale_img = s_img,
                                        mask_threshold = 0.0,
                                        return_heatmap = True)                                         

print(f'Saving edit ({edit_text})')
torchvision.utils.save_image(tensor = torch.cat( ( image_tensor.unsqueeze(0).cpu(), 
                                                   edited_image.unsqueeze(0).cpu(), 
                                                   heatmap.expand(-1,3,-1,-1).cpu(),
                                                   edited_image_ip2p.unsqueeze(0).cpu()),
                                                dim = 0), 
                             normalize = True,
                             scale_each = True,
                             fp = 'imgs/pleural.png')


#