# PPVF_MAINCODE


1. Install the requirements packages following the file "requirements.txt".

2. Creat and rename a python file "global_config.py" with lines as follows:

-----

GLOBAL_PATH = "/your/code/path/"

DATA_PATH =  "/your/datasets/path/"

-----

3. Run the following example code:

python3 PPF_v0.1.py \
--predict_policy HPP_PAC \
--fetch_policy PPVF \
--run_way training \
--train_level edge \
--PPF_update_interval_day 2 \
--dataset_percent 1 \
--c_e_ratio "0.01" \
--f_e "4" \
--epsilon "1.0" \
--xi "15" \
--noise 


