class FeatureTracker:
    def __init__(self):
        self.extractor = None
        self.matcher = None

    def track(self, image_ref, image_cur, kps_ref, kps_cur):
        kps_cur, des_cur = self.extractor(image_cur)
        matches = self.matcher(kps_ref, kps_cur)
        return matches
