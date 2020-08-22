from PIL import Image
from PIL.TiffTags import TAGS
import json

with Image.open('images/4_333.tif') as img:
    meta_dict = {TAGS[key]: img.tag[key] for key in img.tag.keys()}
    with open('exif.json', 'w') as f:
        for line in meta_dict:
            f.write(str(line) + ":" + str(meta_dict[line]) + '\n')
