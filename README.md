# DeepTrader2

This repository holds the code and models used to produce DeepTrader2, an independent replication of the original DeepTrader, as part of my Master's thesis, the results from which contributed to the conference paper:

[Automated Creation of a High-Performing Algorithmic Trader via Deep Learning on Level-2 Limit Order Book Data](https://arxiv.org/abs/2012.00821).

**Models** - Contains the different versions of DeepTrader2 models produced during experimentation

**DeepTrader Versions** - Contains each permutation of each DeepTrader2, the best performing of which was 'DeepTraderv4_combined_filtered_top20_decay_10epochs', which means:
- 4th iteration of DeepTrader2 versions
- combined two seperate datasets 
- filtered the data to only include the top 20 traders
- decay was applied to the learning rate of the model during training
- the model was trained for 10 epochs

**Experiments** - Contains the versions of BSE used during experimentation - a different version was used on the virtual machines to automate results.

**utils** - These were the python files used to automate the AWS sessions and collect the data.
