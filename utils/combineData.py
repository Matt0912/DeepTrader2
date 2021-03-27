import os
import glob
import pandas as pd
os.chdir(r"C:\Users\Matt\Documents\CompSciLinux\Thesis\BristolStockExchange\experiments\data_v4_(schedule_offset)")

extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

firstFile = pd.read_csv(all_filenames[0])
firstFile.to_csv("result.csv", index=False)

result = open('result.csv', 'a')

for i in range(1, len(all_filenames)):
    with open(all_filenames[i], 'r') as f1:
        original = f1.read()
        result.write(original)