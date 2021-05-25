import numpy as np
import os
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torchvision.utils import save_image
from scipy.stats import norm

from inception_score import inception_score

seed = 31415
np.random.seed(seed)
torch.manual_seed(seed)


def evaluate():
    param = {
        "path_data": "../input/celeba/",
        "path_model": "../models/celeba/",
        "path_out": "../output/out_celeba/",
        "path_true": "../output/out_celeba/true_data/",
    }

    names = {
        "vae": "Usual VAE",
        "rockar": "Rockarfellar VAE",
        "ada": "AdaCVaR VAE",
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 64
    n_generated_images = 5000
    n_inception = 1000
    output_file = open(param["path_out"] + "output.txt", "w")

    # Generate true images
    if not os.path.exists(param["path_true"]):
        os.makedirs(param["path_true"])

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    valid_set = datasets.CelebA(
        root=param["path_data"],
        split="valid",
        download=True,
        transform=transform,
    )

    for cur in ["vae", "rockar", "ada"]:
        output_file.write(names[cur] + "\n")
        model = torch.load(
            param["path_model"] + cur, map_location=torch.device("cpu")
        ).to(device)
        path_out_data = param["path_out"] + cur + "_data/"
        path_out = param["path_out"] + cur

        # Generate reconstructed images
        if not os.path.exists(path_out_data):
            os.makedirs(path_out_data)
        rec_data = torch.tensor([])
        with torch.no_grad():
            for i in range(n_generated_images // 128):
                cur_data = model.sample(128, device)
                for j in range(128):
                    save_image(cur_data[j], path_out_data + str(128 * i + j) + ".png")
                rec_data = torch.cat((rec_data, cur_data.cpu()), dim=0)

        # with torch.no_grad():
        # rec_data = model.sample(n_inception, device).cpu().numpy()
        # rec_data = np.repeat(rec_data, 3, 1)
        score, _ = inception_score(
            rec_data,
            batch_size=32,
            resize=True,
            splits=10,
        )
        output_file.write("Inception score: {:.2f} \n".format(score))
        output_file.write("\n")
    output_file.close()


if __name__ == "__main__":
    evaluate()
