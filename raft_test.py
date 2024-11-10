#%%
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.io import decode_image

from torchvision.models.optical_flow import raft_large


plt.rcParams["savefig.bbox"] = "tight"
# sphinx_gallery_thumbnail_number = 2


def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()

def get_frame(path: str, frame: int) -> torch.Tensor:
    raw_name = f"{frame}-10.jpg"
    padded_name = (5 - len(str(frame))) * "0" + raw_name
    name = os.path.join(path, padded_name)
    img = decode_image(name)
    return img

def preprocess(batch):
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            T.Resize(size=(520, 1024)),
        ]
    )
    batch = transforms(batch)
    return batch

#%%
from torchvision.utils import flow_to_image
from torchvision.io import write_jpeg

device = "cuda" if torch.cuda.is_available() else "cpu"
model = raft_large(pretrained=True, progress=False).to(device)
model = model.eval()
output_folder = "flows/"  # Update this to the folder of your choice
for i in range(2, 1908):
    frame1 = get_frame("pNEUMA10/", i - 1)
    frame2 = get_frame("pNEUMA10/", i)

    img1_batch = torch.stack([frame1])
    img2_batch = torch.stack([frame2])
    img1_batch = preprocess(img1_batch).to(device)
    img2_batch = preprocess(img2_batch).to(device)

    list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
    predicted_flow = list_of_flows[-1][0]
    np.save(output_folder + f"raft_frame_{i}.npy", predicted_flow.detach().numpy())
    if i % 100 == 0:
        print(f"Done iteration {i+1}")

#%%

frame1 = decode_image("pNEUMA10/00001-10.jpg")
frame2 = decode_image("pNEUMA10/00002-10.jpg")

frame3 = decode_image("pNEUMA10/00003-10.jpg")
frame4 = decode_image("pNEUMA10/00004-10.jpg")


img1_batch = torch.stack([frame1, frame2])
img2_batch = torch.stack([frame3, frame4])

#%%

# If you can, run this example on a GPU, it will be a lot faster.

img1_batch = preprocess(img1_batch).to(device)
img2_batch = preprocess(img2_batch).to(device)

print(f"shape = {img1_batch.shape}, dtype = {img1_batch.dtype}")
# %%

model = raft_large(pretrained=True, progress=False).to(device)
model = model.eval()

list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
print(f"type = {type(list_of_flows)}")
print(f"length = {len(list_of_flows)} = number of iterations of the model")

# %%
predicted_flows = list_of_flows[-1]
print(f"dtype = {predicted_flows.dtype}")
print(f"shape = {predicted_flows.shape} = (N, 2, H, W)")
print(f"min = {predicted_flows.min()}, max = {predicted_flows.max()}")
# %%
from torchvision.utils import flow_to_image

flow_imgs = flow_to_image(predicted_flows)

# The images have been mapped into [-1, 1] but for plotting we want them in [0, 1]
img1_batch = [(img1 + 1) / 2 for img1 in img1_batch]

grid = [[img1, flow_img] for (img1, flow_img) in zip(img1_batch, flow_imgs)]
plot(grid)
# %%
# %%