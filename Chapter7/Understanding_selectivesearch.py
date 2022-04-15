from torch_snippets import *
import selectivesearch
from skimage.segmentation import felzenszwalb

def extract_candidates(img):
        img_lbl, regions = selectivesearch.selective_search(img, \
                                     scale=200, min_size=100)
        img_area = np.prod(img.shape[:2])
        candidates = []
        for r in regions:
            if r['rect'] in candidates: continue
            if r['size'] < (0.05*img_area): continue
            if r['size'] > (1*img_area): continue
            x, y, w, h = r['rect']
            candidates.append(list(r['rect']))
        return candidates

img = read('/Users/stefana/Documents/P1_Facial_Keypoints/data/Hemanvi.jpeg', 1)
candidates = extract_candidates(img)
show(img, bbs=candidates)