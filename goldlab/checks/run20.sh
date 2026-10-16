cd /home/claude/code/goldlab_checks
until [ -f p16_done.flag ]; do sleep 5; done
L=20 C=exact MEAS=strip M1=0.6 M2=1.2 CAP=200 python3 -u pair_c2.py > p20_strip_0612.log 2>&1
echo ВСЁ > p20_done.flag
