cd /home/claude/code/goldlab_checks
L=24 DL=0.3 C=exact MEAS=count CAP=300 python3 -u pair_c2.py > pL24_c03.log 2>&1
L=28 DL=0.3 C=exact MEAS=count CAP=300 python3 -u pair_c2.py > pL28_c03.log 2>&1
echo ВСЁ > pL_done.flag
