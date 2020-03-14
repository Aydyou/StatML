# Welcome to GitHub Desktop!

This project includes some components of sentiment analysis, question answering and translation. When Main.py is run and open the localhost on your browser, you'll see three sections.
First section if related to sentiments analysis. You can enter Yelp reviews and it will tell you whether it is positive or negative. If it is negative it will tell you what is the reason. It does support both English and German languages.
Second section measures the relevance of the two senteces entered. The third section finds the answer of the question (first box), in the text of the second box. All sections support English and German.



In order to use pretrained data please download the following files and place them in their respective directories.

If you want to train the model, you need a csv file with name "train.csv" in the data folder. Furthermore you will need to uncomment all codes in blocks numbered 1,3,5 19 and 22 in Main.ipynb and also turn the 'do_train' to True in args, located at block 8.

After training and generating the files you can return everything to the current status and run it.


Before running you would need to install the following packages (You definitely need Pytorch):

`pip install transformers`
`pip install fairseq`
`pip install fastBPE`

If you are on a windows machine for the second and third command you will get error. Some parts of fairseq will be installed but not all parts. The part we need is fastBPE. Installing this on a windows machine is slightly challenging. You can follow <https://github.com/pytorch/fairseq/issues/1224> to make it work. For simplicity I have added the repo fastBPE which also includes the files mman.h and mman.c and the changes required. You'll need to follow from <https://github.com/pytorch/fairseq/issues/1224#issuecomment-539562932> onwards.

Running this for the first time will take a little bit of time. It will start downloading some pretrained models, for example for translation (fairseq). Depending on your internet speed it can take over and hour for completion (Even if you have downloaded all the data linked above.)




