cd /home/claude/code/goldlab_checks
while [ ! -f pD_done.flag ]; do sleep 5; done
L=16 DL=0.3 C=exact MEAS=count CAP=300 python3 -u pair_c2.py > pC16_03.log 2>&1
L=16 DL=0.6 C=exact MEAS=count CAP=300 python3 -u pair_c2.py > pC16_06.log 2>&1
L=18 DL=0.6 C=exact MEAS=count CAP=300 python3 -u pair_c2.py > pC18_06.log 2>&1
L=18 DL=0.3 C=exact MEAS=count CAP=300 python3 -u pair_c2.py > pC18_03.log 2>&1
echo ВСЁ > pC_done.flag
