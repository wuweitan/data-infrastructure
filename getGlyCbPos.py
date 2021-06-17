from numpy import pi
from Bio.PDB.vectors import rotaxis2m
from Bio.PDB.vectors import Vector
import numpy as np
from Bio.PDB import *


def getGlyCbPos(residue):
	"""
	Calculate the C Beta position for the Glycine

	Input:
	1. residue (residue object for the Glycine)

	Output:
	1. cb (The position coordinates vector for C Beta)

	"""
	
	# check if n/c/ca coordinate info exists. (If no, return None)
	try:
		# get atom coordinates as vectors
		n = residue["N"].get_vector()
		c = residue["C"].get_vector()
		ca = residue["CA"].get_vector()
	except Exception:
		return None

	# center at origin
	n = n - ca
	c = c - ca
	# find rotation matrix that rotates n -120 degrees along the ca-c vector
	rot = rotaxis2m(-pi * 120.0 / 180.0, c)
	# apply rotation to ca-n vector
	cb_at_origin = n.left_multiply(rot)
	# put on top of ca atom
	cb = cb_at_origin + ca

	return cb