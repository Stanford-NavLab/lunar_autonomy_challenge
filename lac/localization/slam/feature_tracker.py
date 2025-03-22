"""SuperPoint+LightGlue-based feature tracker"""

from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd


class FeatureTracker:
    def __init__(self):
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()
        self.matcher = LightGlue(features="superpoint").eval().cuda()

    def track(self, image_ref, image_cur, kps_ref, kps_cur):
        kps_cur, des_cur = self.extractor(image_cur)
        matches = self.matcher(kps_ref, kps_cur)
        return matches
