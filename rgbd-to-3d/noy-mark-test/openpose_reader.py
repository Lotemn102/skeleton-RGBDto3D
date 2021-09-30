import glob
import json

class OpenPoseObject(object):
    def __init__(self, image_id, keypoints):
        self.image_id = image_id + ".png"
        self.keypoints = keypoints

    def __eq__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return self.image_id == other.image_id

    def __hash__(self):
        return self.image_id.__hash__()

    # object to string
    def __repr__(self):
        return "image id: " + str(self.image_id)

    def __eq__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return self.image_id == other.image_id

    def __hash__(self):
        return (self.image_id).__hash__()

def OpenPoseReader(path, orientation):
    # final objects list
    openpose_list = []
    # original path to files
    # raw_filenames = glob.glob('*openpose/*.json')
    raw_filenames = glob.glob(path + orientation + '_openpose/*.json')
    # constructing filenames
    for item in raw_filenames:
        # open json file
        with open(item) as json_file:
            data = json.load(json_file)
        name_temp = item.replace(path + orientation + '_openpose\\', '')
        # get image id name
        image_id = name_temp.replace('_keypoints.json', '')
        # get keypoints
        if len(data['people']) > 0:
            keypoints = data['people'][0]['pose_keypoints_2d']
            openpose_list.append(OpenPoseObject(image_id, keypoints))
    return openpose_list