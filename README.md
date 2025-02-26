# Geo SSDG
<p>This project is built upon <strong>Dassl.pytorch</strong> and <strong>ssdg-benchmark</strong>. To set up the environment and prepare the datasets, please follow the instructions provided in the repositories:</p>

<ul>
    <li><a href="https://github.com/KaiyangZhou/Dassl.pytorch">Dassl.pytorch</a></li>
    <li><a href="https://github.com/KaiyangZhou/ssdg-benchmark">ssdg-benchmark</a></li>
</ul>

<h1>How to Run</h1>

<p>You can find the script in <code>/scripts/FBASA/run_ssdg.sh</code>. Ensure that you update the <code>DATA</code> variable to point to the directory where your datasets are stored. The script accepts two input arguments:</p>

<ul>
    <li><code>DATASET</code> – The name of the dataset.</li>
    <li><code>NLAB</code> – The total number of labels.</li>
</ul>

<p>For example, if you want to execute <strong>FBC-SA</strong> on <strong>OfficeHome</strong> under the 10-labels-per-class configuration (which results in a total of 1950 labels), run the following commands in your terminal:</p>

<pre><code>
conda activate dassl
cd scripts/FBCSA
bash run_ssdg.sh ssdg_pacs 210
</code></pre>

<p>In this setup, the code will execute <strong>FBC-SA</strong> across four different target domains, each running five times using different random seeds. If you have multiple GPUs and prefer running a single experiment instead of all at once, you can modify the script accordingly.</p>

<p>To display the results, run:</p>

<pre><code>
python parse_test_res.py output/ssdg_officehome/nlab_1950/FBCSA/resnet18 --multi-exp
</code></pre>
