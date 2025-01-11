import torch
import torchvision.transforms.functional as torchvision_F
import numpy as np
import os
import sys
import shutil
import importlib
import cv2
import utils.options as options
import utils.util_vis as util_vis

from utils.util import EasyDict as edict
from PIL import Image
from utils.eval_3D import get_dense_3D_grid, compute_level_grid, convert_to_explicit
from tqdm import tqdm

def get_1d_bounds(arr):
    nz = np.flatnonzero(arr)
    return nz[0], nz[-1]

def get_bbox_from_mask(mask, thr):
    """Compute bounding box [x0, y0, x1, y1] from a binary mask."""
    masks_for_box = (mask > thr).astype(np.float32)
    assert masks_for_box.sum() > 0, "Empty mask!"
    x0, x1 = get_1d_bounds(masks_for_box.sum(axis=-2))
    y0, y1 = get_1d_bounds(masks_for_box.sum(axis=-1))

    return x0, y0, x1, y1

def square_crop(image, bbox, crop_ratio=1.0):
    """Crop the image to a square region around the bounding box."""
    x1, y1, x2, y2 = bbox
    h, w = y2 - y1, x2 - x1
    yc, xc = (y1 + y2) / 2, (x1 + x2) / 2
    S = max(h, w) * 1.2  # expand a bit
    scale = S * crop_ratio
    image = torchvision_F.crop(
        image,
        top=int(yc - scale / 2),
        left=int(xc - scale / 2),
        height=int(scale),
        width=int(scale),
    )
    return image

def preprocess_image(opt, image, bbox):
    """
    Preprocess the RGBA image by:
    1. Cropping to a square around `bbox`.
    2. Resizing if necessary.
    3. Splitting into rgb, mask (from alpha channel).
    4. Optionally replacing the background with `opt.data.bgcolor`.
    """
    # Crop
    image = square_crop(image, bbox=bbox)
    # Resize
    if image.size[0] != opt.W or image.size[1] != opt.H:
        image = image.resize((opt.W, opt.H))
    # Convert to Tensor
    image = torchvision_F.to_tensor(image)  # [4, H, W]
    rgb, mask = image[:3], image[3:]
    # If a background color is specified, blend
    if opt.data.bgcolor is not None:
        # alpha-based blending
        rgb = rgb * mask + opt.data.bgcolor * (1 - mask)
        # Binarize the mask
        mask = (mask > 0.5).float()
    return rgb, mask

def get_image(opt, image_name):
    """
    Read a single RGBA image from `opt.datadir/image_name`.
    Extract bounding box from alpha channel, then call `preprocess_image`.
    """
    rgba_path = os.path.join(opt.datadir, image_name)
    rgba = Image.open(rgba_path).convert("RGBA")  # Ensures 4 channels: RGBA

    # Extract alpha as mask
    alpha_np = np.array(rgba)[:, :, -1]  # shape [H, W]
    # Binarize alpha
    alpha_np[alpha_np <= 127] = 0
    alpha_np[alpha_np > 127] = 1.0

    # Compute bounding box from alpha channel
    bbox = get_bbox_from_mask(alpha_np, thr=0.5)

    # Preprocess
    rgb_input_map, mask_input_map = preprocess_image(opt, rgba, bbox=bbox)
    return rgb_input_map, mask_input_map

def erode_mask(mask, iterations=5):
    """Erode the mask by some iterations."""
    mask_np = mask.squeeze(0).cpu().numpy().astype(np.uint8)
    mask_eroded = cv2.erode(mask_np, np.ones((3,3), np.uint8), iterations=iterations)
    mask_eroded = torch.tensor(mask_eroded).unsqueeze(0).float()
    return mask_eroded

def get_intr(opt):
    """Simple intrinsic matrix generator."""
    f = 1.3875
    K = torch.tensor([
        [f * opt.W, 0,         opt.W / 2],
        [0,         f * opt.H, opt.H / 2],
        [0,         0,         1],
    ]).float()
    return K

def get_pixel_grid(H, W, device='cuda'):
    y_range = torch.arange(H, dtype=torch.float32).to(device)
    x_range = torch.arange(W, dtype=torch.float32).to(device)
    Y, X = torch.meshgrid(y_range, x_range, indexing='ij')
    Z = torch.ones_like(Y).to(device)
    xyz_grid = torch.stack([X, Y, Z], dim=-1).view(-1, 3)
    return xyz_grid

def unproj_depth(depth, intr):
    """
    depth: [B, H, W]
    intr: [B, 3, 3]
    """
    batch_size, H, W = depth.shape
    intr = intr.to(depth.device)
    
    K_inv = torch.linalg.inv(intr).float()  # [B, 3, 3]
    pixel_grid = get_pixel_grid(H, W, depth.device).unsqueeze(0)  # [1, H*W, 3]
    pixel_grid = pixel_grid.repeat(batch_size, 1, 1)              # [B, H*W, 3]
    ray_dirs = K_inv @ pixel_grid.permute(0, 2, 1).contiguous()   # [B, 3, H*W]
    seen_points = ray_dirs.permute(0, 2, 1).contiguous() * depth.view(batch_size, H * W, 1)
    seen_points = seen_points.view(batch_size, H, W, 3)
    return seen_points

def prepare_data(opt):
    """
    Prepare data from a single directory `opt.datadir` of RGBA images.
    """
    datadir = opt.datadir
    # Collect RGBA images (png/jpg)
    image_names = [
        name for name in os.listdir(datadir)
        if name.lower().endswith(".png") or name.lower().endswith(".jpg")
    ]
    image_names.sort()
    
    data_list = []
    for i, image_name in enumerate(image_names):
        rgb_input_map, mask_input_map = get_image(opt, image_name)
        intr = get_intr(opt)
        
        var = edict()
        var.rgb_input_map = rgb_input_map.unsqueeze(0).to(opt.device)
        var.mask_input_map = mask_input_map.unsqueeze(0).to(opt.device)
        var.intr = intr.unsqueeze(0).to(opt.device)
        var.idx = torch.tensor([i + 1]).to(opt.device).long()
        var.pose_gt = False
        
        if opt.task == 'depth':
            var.mask_eroded = erode_mask(mask_input_map.squeeze(0), iterations=4)
            var.mask_eroded = var.mask_eroded.view_as(mask_input_map).to(opt.device)

        data_list.append(var)

    # strip file extension for nice naming
    name_list = [os.path.splitext(name)[0] for name in image_names]
    return data_list, name_list

@torch.no_grad()
def marching_cubes(opt, var, impl_network, visualize_attn=False):
    points_3D = get_dense_3D_grid(opt, var)  # [B, N, N, N, 3]
    level_vox, attn_vis = compute_level_grid(
        opt, impl_network, var.latent_depth, var.latent_semantic,
        points_3D, var.rgb_input_map, visualize_attn
    )
    if attn_vis:
        var.attn_vis = attn_vis
    level_grids = level_vox.cpu().numpy()
    meshes = convert_to_explicit(opt, level_grids, isoval=0.5, to_pointcloud=False)
    var.mesh_pred = meshes
    return var

def main():
    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd, safe_check=False)
    opt.device = 0
    
    # build model
    task_ckpt = opt.yaml.split('/')[-1].split('.')[0]
    if task_ckpt != opt.task:
        raise ValueError('Detected different tasks between specified and the yaml, please double check!')
    if opt.task == 'shape':
        opt.pretrain.depth = None
    opt.arch.depth.pretrained = None
    module = importlib.import_module("model.compute_graph.graph_{}".format(opt.task))
    graph = module.Graph(opt).to(opt.device)
    
    # load checkpoint
    checkpoint = torch.load(opt.ckpt, map_location=torch.device(opt.device))
    ep, it, best_val = checkpoint["epoch"], checkpoint["iter"], checkpoint["best_val"]
    print(f"resuming from epoch {ep + 1} (iteration {it}, best_val {best_val:.4f})")
    graph.load_state_dict(checkpoint["graph"], strict=True)
    graph.eval()
    print('==> checkpoint loaded')
    
    # load the data
    data_list, name_list = prepare_data(opt)
    print(f'==> sample data loaded from folder: {opt.datadir}')
    
    # create the save dir
    save_folder = os.path.join(opt.datadir, 'preds')
    if os.path.isdir(save_folder):
        shutil.rmtree(save_folder)
    os.makedirs(save_folder)
    opt.output_path = opt.datadir
    
    # inference the model and save the results
    progress_bar = tqdm(data_list)
    for i, var in enumerate(progress_bar):
        # forward
        with torch.no_grad():
            var = graph.forward(opt, var, training=False, get_loss=False)
            
            if opt.task == 'shape':
                var = marching_cubes(opt, var, graph.impl_network, visualize_attn=True)
                # save results
                util_vis.dump_images(
                    opt, [name_list[i]], "image_input", var.rgb_input_map,
                    masks=None, from_range=(0, 1), folder='preds'
                )
                util_vis.dump_images(
                    opt, [name_list[i]], "mask_input", var.mask_input_map,
                    folder='preds'
                )
                util_vis.dump_attentions(
                    opt, [name_list[i]], "attn", var.attn_vis, folder='preds'
                )
                util_vis.dump_meshes(
                    opt, [name_list[i]], "mesh", var.mesh_pred, folder='preds'
                )
                util_vis.dump_meshes_viz(
                    opt, [name_list[i]], "mesh_viz", var.mesh_pred,
                    save_frames=False, folder='preds'
                )  # image frames + gifs

            elif opt.task == 'depth':
                # [B, H, W, 3]
                seen_surface_fixed = unproj_depth(var.depth_pred.squeeze(1), var.intr)
                seen_surface_pred = unproj_depth(var.depth_pred.squeeze(1), var.intr_pred)
                validity_mask = var.mask_input_map.view(
                    seen_surface_pred.shape[0],
                    seen_surface_pred.shape[1],
                    seen_surface_pred.shape[2], 1
                )
                seen_surface_fixed = seen_surface_fixed * validity_mask + (1 - validity_mask) * -1
                seen_surface_pred = seen_surface_pred * validity_mask + (1 - validity_mask) * -1

                util_vis.dump_images(
                    opt, [name_list[i]], "image_input", var.rgb_input_map,
                    masks=None, from_range=(0, 1), folder='preds'
                )
                util_vis.dump_images(
                    opt, [name_list[i]], "mask_input", var.mask_input_map,
                    folder='preds'
                )
                util_vis.dump_depths(
                    opt, [name_list[i]], "depth_est", var.depth_pred,
                    var.mask_input_map, rescale=True, folder='preds'
                )
                util_vis.dump_seen_surface(
                    opt, [name_list[i]], "seen_surface_fixed", "image_input",
                    seen_surface_fixed, folder='preds'
                )
                util_vis.dump_seen_surface(
                    opt, [name_list[i]], "seen_surface_pred", "image_input",
                    seen_surface_pred, folder='preds'
                )
                
    print(f'==> results saved at folder: {opt.datadir}/preds')

if __name__ == "__main__":
    main()
