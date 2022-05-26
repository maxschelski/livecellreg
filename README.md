# livecellreg
Correlation-based registration of 2D &amp; 3D live cell imaging movies using translation of images.

Still needs a lot of refactoring. Will be converted in a locally installable package in June-July 2022.

For now do the following to use the script
1) execute script livecellreg.py with 
- choose_folder_manually=True and 
-  for multi channel movies set the channel on which the registration should be based as reference_channel = "c000X" where X is the channel number (counting starts at 0). 
2) A popup window will open in which you can choose in which folder you want to register all tiff files.

For questions, please write to me at max.schelski@googlemail.com.
