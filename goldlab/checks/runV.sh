cd /home/claude/code/goldlab_checks
L=16 C=exact MEAS=voronoi CAP=300 python3 -u pair_c2.py > pV16_11.log 2>&1
L=18 C=exact MEAS=voronoi CAP=300 python3 -u pair_c2.py > pV18_11.log 2>&1
echo ВСЁ > pV_done.flag
