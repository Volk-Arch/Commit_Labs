cd /home/claude/code/goldlab_checks
L=16 C=exact MEAS=strip python3 -u pair_c2.py > p16_strip_11.log 2>&1
L=16 C=exact MEAS=strip M1=0.6 M2=1.2 python3 -u pair_c2.py > p16_strip_0612.log 2>&1
L=16 C=exact MEAS=level CAP=300 python3 -u pair_c2.py > p16_level_11.log 2>&1
echo ВСЁ > p16_done.flag
