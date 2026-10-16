cd /home/claude/code/goldlab_checks
until [ -f p20_done.flag ]; do sleep 5; done
L=18 C=exact MEAS=level CAP=300 python3 -u pair_c2.py > p18_level_11.log 2>&1
echo ВСЁ > p18_done.flag
