import numpy as np
from prody import *
import re, sys

class Pfam_sequence():
  """Pfam sequence module
     Provides functions to process pfam sequence data. e.g. MSA in 'selex', 'stockholm' and 'fasta' format
     Currently only support processing 'stockholm' format, others will be added later.

     Functions:

       * count_statis():
         -

       * 
  """
  def __init__(self, data_dir, **kwargs):
    self.data_dir
    #self.arg1 = kwargs.pop('arg1', None)


