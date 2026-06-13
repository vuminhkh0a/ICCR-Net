cd ~
cd KhoaVM
source khoa-env/bin/activate
cd Proposed_Contrastive3
nohup bash -c '
python3 "[OTU2D]-SimCLR.py" &&
python3 "[OTU2D]-Moco.py" &&
python3 "[OTU2D]-BYOL.py" &&
python3 "[OTU2D]-simsiam.py"
' > log2.txt 2>&1 &
tail -f log2.txt