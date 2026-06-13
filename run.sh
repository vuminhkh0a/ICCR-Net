cd ~
cd KhoaVM
source khoa-env/bin/activate
cd Proposed_Contrastive3
nohup python3 main.py > log.txt 2>&1 &
tail -f log.txt