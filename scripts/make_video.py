from glob import glob
from typing import Union, Tuple, List
import cv2
import os
from tqdm import tqdm
import zipfile


def images_to_video(images_files: List[str],
                    output: str,
                    center_crop: Union[Tuple[int, int], None] = None,
                    resize: Union[Tuple[int, int], None] = None,
                    fps: int = 30,
                    zip_input: bool = False):
    '''
    Convert a sequence of images into a video
    
    images_files: List of images path, order-sensitive
    output: Video file to be write, NO POSTFIX ! 
    center_crop: None or tuple[int, int], whether crop the center area of videos
    final_size: None or tuple[int, int], the H and W of final video. Image resize may applied
    fps: Frames per second of final video
    zip_input: Whether move all input files into a zip
    
    '''
    try: 
        f_0 = images_files[0]
        h, w, c = cv2.imread(f_0).shape

        if (resize is None):
            if (center_crop is None):
                final_size = (h, w)
            else:
                final_size = center_crop
        else:
            final_size = resize

        video = cv2.VideoWriter(output + '.mp4', cv2.VideoWriter_fourcc(*'x264'), fps, final_size)
        for file in tqdm(images_files, desc=f'Compressing Video: {os.path.basename(output+".mp4")}', dynamic_ncols=True):
            img = cv2.imread(file)
            if (center_crop is not None):
                h_new, w_new = center_crop
                img = img[h // 2 - h_new // 2:h // 2 + h_new // 2,
                        w // 2 - w_new // 2:w // 2 + w_new // 2, :]  #  center crop
            if (resize is not None):
                img = cv2.resize(img, resize)
            video.write(img)
        video.release()

        if (zip_input == True):
            zip = zipfile.ZipFile(
                os.path.join(output + '.zip'),
                'w',
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
            for f in tqdm(images_files, desc=f'Compressing Images: {os.path.basename(output+".zip")}', dynamic_ncols=True):
                zip.write(f, os.path.basename(f))
                os.remove(f)
            zip.close()
    except:
        pass
