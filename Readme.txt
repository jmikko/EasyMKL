~ These files are distributed under the "GNU GENERAL PUBLIC LICENSE" (contained in file LICENSE).

~ Authors: Michele Donini and Fabio Aiolli

~ Citation Request:
Use of this code in publications should be acknowledged by referencing the following publication:
"EasyMKL: a scalable multiple kernel learning algorithm by Fabio Aiolli and Michele Donini, Neurocomputing, 2015"
@article{aiolli2015easymkl,
  title={EasyMKL: a scalable multiple kernel learning algorithm},
  author={Aiolli, Fabio and Donini, Michele},
  journal={Neurocomputing},
  year={2015},
  publisher={Elsevier}
}

~ Site: http://www.math.unipd.it/~mdonini/publications.html



= = = Python Folder = = =
~ Required Python packages
	numpy
	scikit-learn
	cvxopt

----------------------------------------------------------------------------------------------------
File			Content

EasyMKL.py		EasyMKL implementation
komd.py			Scikit-like implementation of the kernel machine KOMD

toytest_EasyMKL.py	An example of EasyMKL
toytest_komd.py		An example of KOMD

komd_gui.py		Graphical toytest interface for KOMD





= = = R Folder = = =
~ Required R package
	kernlab

----------------------------------------------------------------------------------------------------
File			Content

komd.R			R implementation of the kernel machine KOMD with a toy example
