MINT (Mesoscale TINT)
====
This is a fork of the base TINT repository, intended to be used for tracking
and analysing large mesoscale objects in radar reflectivity images. Modifications made by Ewan Short while undertaking a PhD at the University of Melbourne.

Below are some animations for example systems identified in the CPOL research radar record.
Click on an animation to view at native resolution, or right click "save image as" to download.

Example front fed trailing stratiform system.
![MINT](FFTS.gif "Demo")

Example front fed leading stratiform system.
![MINT](FFLS.gif "Demo")

Example rear fed trailing stratiform system.
![MINT](RFTS.gif "Demo")

Dependencies
------------
- NumPy
- Pandas
- SciPy
- matplotlib
- cartopy
- Py-ART
- ffmpeg

Install
-------
To install TINT, first install the dependencies listed above. We recommend
installing Py-ART from conda forge::

	conda install -c conda-forge arm_pyart

Then clone::

	git clone https://github.com/openradar/TINT.git

then::

	cd TINT
	python setup.py install

Acknowledgements
----------------
This work is the adaptation of tracking code in R created by Bhupendra Raut who was working at Monash University,
Australia in the Australian Research Council's Centre of Excellence for Climate System Science led by Christian Jakob.
This work was supported by the Department of Energy, Atmospheric Systems Research (ASR) under Grant DE-SC0014063,
“The vertical structure of convective mass-flux derived from modern radar systems - Data analysis in support of cumulus
parametrization”

The development of this software is supported by the Climate Model Development
and Validation (CMDV) activity which funded by the Office of Biological and
Environmental Research in the US Department of Energy Office of Science.

References
----------
Dixon, M. and G. Wiener, 1993: TITAN: Thunderstorm Identification, Tracking,
Analysis, and Nowcasting—A Radar-based Methodology. J. Atmos. Oceanic
Technol., 10, 785–797, doi: 10.1175/1520-0426(1993)010<0785:TTITAA>2.0.CO;2.

Leese, J.A., C.S. Novak, and B.B. Clark, 1971: An Automated Technique for Obtaining Cloud Motion from Geosynchronous
Satellite Data Using Cross Correlation. J. Appl. Meteor., 10, 118–132, doi: 10.1175/1520-0450(1971)010<0118:AATFOC>2.0.CO;2.
