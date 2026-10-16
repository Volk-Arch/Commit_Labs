cd /home/claude/code/goldlab_checks
L=16 DL=0.6 C=exact MEAS=strip CAP=300 python3 -u pair_c2.py > pD16_06.log 2>&1
L=16 DL=0.9 C=exact MEAS=strip CAP=300 python3 -u pair_c2.py > pD16_09.log 2>&1
L=18 DL=0.9 C=exact MEAS=strip CAP=300 python3 -u pair_c2.py > pD18_09.log 2>&1
echo ВСЁ > pD_done.flag
