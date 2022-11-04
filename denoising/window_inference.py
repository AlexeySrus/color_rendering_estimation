import torch


def inference_tiling_intersected(
        img: torch.Tensor, single_inference: callable, tile_size=256, stride_k=1) -> torch.Tensor:
    """
    Process the image with splitting on tiles.
    `singel_inferece` will be applied to each tile. Its expected the input
    image is torch.Tensor [C, H, W] shape of float type.
    """
    res_mask = torch.zeros(img.size(0), img.size(1), img.size(2), dtype=torch.float32, device=img.device)
    counter_mask = torch.zeros(img.size(1), img.size(2), dtype=torch.long, device=img.device)

    stride = tile_size // stride_k

    x0_vec = []
    y0_vec = []

    target_x = 0
    while target_x + tile_size < img.size(2):
        x0_vec.append(target_x)
        target_x += stride
    x0_vec.append(img.size(2) - tile_size - 1)

    target_y = 0
    while target_y + tile_size < img.size(1):
        y0_vec.append(target_y)
        target_y += stride
    y0_vec.append(img.size(1) - tile_size - 1)

    for y0 in y0_vec:
        for x0 in x0_vec:
            img_crop = img[:, y0:y0 + tile_size, x0:x0 + tile_size]
            res = single_inference(img_crop.unsqueeze(0)).squeeze(0)
            res_mask[y0:y0 + tile_size, x0:x0 + tile_size] += res
            counter_mask[y0:y0 + tile_size, x0:x0 + tile_size] += 1

    return res_mask / counter_mask
