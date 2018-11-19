This footwear impression database contains original crime scene impressions and a database of
reference impressions. This data is limited to non-commercial use only.

If you use this database in your research please cite:

Unsupervised Footwear Impression Analysis and Retrieval from Crime Scene Data,
Adam Kortylewski, Thomas Albrecht, Thomas Vetter. ACCV 2014, Workshop on Robust Local Descriptors


Directories and files included in the database:

references/		- Reference impressions; scaled to the mean height of 586 pixels,
			corresponding to approx. 20 pixel per centimeter
tracks_original/ 	- Original police images
tracks_cropped/		- Crime scene impressions are cropped in a way that they are roughly 
			centered in the image frame; scaled to 20 pixel per centimeter
label_table.(csv|mat)	- Number of the correct reference for each track
period_label.(csv|mat)	- Tracks that have been labeled as periodic by our 			  
			algorithm. These 133 have been used during our experiments.

---------------
Contact

adam dot kortylewski at unibas dot ch

---------------
Acknowledgments

We thank the German State Criminal Police Offices and forensity AG for providing the data and
supporting its publication. 






