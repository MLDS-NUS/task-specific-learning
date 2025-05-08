# README
This is the code for the numerical example of *Multistep prediction of dynamical systems*

To generate data, train with MSE, and train with TS
```bash
bash examples/Eg1/E1_run.sh
```
Data can also be downloaded from huggingface using (for all examples)
```bash
python examples/dldata.py
```

To conduct ablation study (generate data, train with MSE, and train with TS)
```bash
bash examples/Eg1/E1_run_as.sh
```
Data can also be downloaded from huggingface 

To consider sample-reweighting methods
```bash
bash examples/Eg1/E1_run_re.sh
```

To determine kernel variance, run 
```bash
python examples/Eg1/code/E1_ker.py
```

To show the numerical results, run examples/Eg1/show/E1_show.ipynb

