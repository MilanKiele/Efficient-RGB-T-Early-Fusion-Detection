import os
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import cv2
import numpy as np

def show_eme_dets(batch_inputs, eme_results_list, save_path=None, save_split=True):
    bs, c, h, w = batch_inputs.shape
    rgbt_np = make_grid(batch_inputs.reshape(-1, 3, h, w), 2, normalize=True, scale_each=True).permute(1,2,0).detach().cpu().numpy()
    rgbi, ti = np.hsplit(rgbt_np, 2)
    rgbi, ti = cv2.cvtColor(rgbi, cv2.COLOR_RGB2BGR), cv2.cvtColor(ti, cv2.COLOR_RGB2BGR)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (255, 0, 255), (255, 128, 0)]
    for idx, box in enumerate(eme_results_list[0].bboxes):
        x0, y0, x1, y1 = list(map(lambda x: x.detach().cpu().numpy(), box))
        cv2.rectangle(rgbi, (int(x0), int(y0)), (int(x1), int(y1)), colors[eme_results_list[0].labels[idx]], 1)
        cv2.rectangle(ti, (int(x0), int(y0)), (int(x1), int(y1)), colors[eme_results_list[0].labels[idx]], 1)
    rgbi, ti = cv2.cvtColor(rgbi, cv2.COLOR_BGR2RGB), cv2.cvtColor(ti, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(np.hstack([rgbi, ti]))
    plt.show()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        rgbi, ti = cv2.cvtColor(rgbi * 255., cv2.COLOR_RGB2BGR), cv2.cvtColor(ti * 255., cv2.COLOR_RGB2BGR)
        assert save_path.endswith('.png')
        if not save_split:
            cv2.imwrite(save_path, np.hstack([rgbi, ti]))
            print(f'images are successfully saved at: {save_path}')
        else:
            cv2.imwrite(save_path.split('.png')[0] + '_rgb.png', rgbi)
            print(f'images (rgb) are successfully saved at: {save_path.split(".png")[0] + "_rgb.png"}')
            cv2.imwrite(save_path.split('.png')[0] + '_infrared.png', ti)
            print(f'images (infrared) are successfully saved at: {save_path.split(".png")[0] + "_infrared.png"}')
