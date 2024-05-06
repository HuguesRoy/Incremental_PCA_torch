# coding: utf8
from clinicadl.utils.caps_dataset.caps_dataset_refactoring.caps_dataset import CapsDatasetImage
from typing import Dict, Optional,Callable,Any
from pathlib import Path
import torch.nn.functional as F
import torch

class CapsDatasetImagePCA(CapsDatasetImage):

    def __init__(
    self,
    caps_directory: Path,
    tsv_label: Path,
    preprocessing_dict: Dict[str, Any],
    train_transformations: Optional[Callable] = None,
    label_presence: bool = True,
    label: str = None,
    label_code: Dict[str, int] = None,
    all_transformations: Optional[Callable] = None
    ):
        """
        Args:
        caps_directory: Directory of all the images.
        data_file: Path to the tsv file or DataFrame containing the subject/session list.
        preprocessing_dict: preprocessing dict contained in the JSON file of prepare_data.
        train_transformations: Optional transform to be applied only on training mode.
        label_presence: If True the diagnosis will be extracted from the given DataFrame.
        label: Name of the column in data_df containing the label.
        label_code: label code that links the output node number to label value.
        all_transformations: Optional transform to be applied during training and evaluation.
        multi_cohort: If True caps_directory is the path to a TSV file linking cohort names and paths.
        """
    
        super(CapsDatasetImagePCA,self).__init__(
            caps_directory,
            tsv_label,
            preprocessing_dict,
            train_transformations,
            label_presence,
            label,
            label_code,
            all_transformations
        )

    @property
    def elem_index(self):
        return None

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        image = sample['image']
        image = torch.flatten(image)

        #return flatten image instead of dictionary because suppose the IncrementalPCA take into account n_sample * n_features 
        return image

    def num_elem_per_image(self):
        return 1

    def size_elem(self):
        return self[0].size()
    
####### transform in my case #########

class MinMaxNormalization(object):
    """ Min Max Normalization."""

    def __call__(self, image : torch.Tensor) -> torch.Tensor:
        image[image != image] = 0                                   
        image = (image - image.min()) / (image.max() - image.min())
        return image


class ResizeInterpolation(object):
  """ Interpolation  """
   
  def __init__(self, size: tuple, mode : str = 'trilinear', align_corners : bool = False):
      self.size = size
      self.mode = mode
      self.align_corners = align_corners

  def __call__(self,image : torch.Tensor) -> torch.Tensor:
    image  = image.unsqueeze(0)
    image = F.interpolate(image, size= self.size, mode= self.mode, align_corners= self.align_corners).squeeze(0)
    return image

class ThresholdNormalization(object):

    def __call__(self,image : torch.Tensor) -> torch.Tensor:
        values, counts = torch.unique(torch.round(image, decimals = 2), return_counts = True)
        threshold = values[counts.argmax()]
        image[image< threshold] = threshold
        return image

class FlattenWithMask(object):
    """ Flatten tensor of the voxels inside the mask"""

    def __init__(self, atlas_path : Path ):
        mask_template =torch.ones((1,1,169,208,179))
        mask_brain = torch.load(atlas_path)
        mask_template[~mask_brain] = 0
        mask_template = F.interpolate(mask_template, size=(128, 128, 128), mode='trilinear', align_corners=False)
        self.mask_template = mask_template >0

    def __call__(self,image: torch.Tensor) -> torch.Tensor:
       return image.unsqueeze(0)[self.mask_template].squeeze(0)
