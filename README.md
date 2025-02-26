# geo_ssdg
This code is built on top of Dassl.pytorch and ssdg-benchmark. Please follow the instructions provided in <a href=https://github.com/KaiyangZhou/Dassl.pytorch>Dassl</a> and <a href=https://github.com/KaiyangZhou/ssdg-benchmark>ssdg-benchmark</a> to install the dassl environment, as well as to prepare the datasets.
<h1> How to run </h1>
The script is provided in /scripts/FBASA/run_ssdg.sh. You need to update the DATA variable that points to the directory where you put the datasets. There are two input arguments: DATASET and NLAB (total number of labels).

Here we give an example. Say you want to run FBC-SA on OfficHome under the 10-labels-per-class setting (i.e. 1950 labels in total), simply run the following commands in your terminal,
```
conda activate dassl
cd scripts/FBCSA
bash run_ssdg.sh ssdg_pacs 210
```
In this case, the code will run FBC-SA in four different setups (four target domains), each for five times (five random seeds). You can modify the code to run a single experiment instead of all at once if you have multiple GPUs.

To show the results, simply do
```
python parse_test_res.py output/ssdg_officehome/nlab_1950/FBCSA/resnet18 --multi-exp

```

