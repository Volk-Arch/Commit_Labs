cd /home/claude/code/goldlab_checks
while pgrep -f "pair_c2.py" > /dev/null; do sleep 5; done
for LL in 16 18 20; do L=$LL C=exact MEAS=voronoi CAP=300 python3 -u pair_c2.py > pVs${LL}.log 2>&1; done
for LL in 16 18; do L=$LL C=exact MEAS=strip CAP=300 python3 -u pair_c2.py > pSs${LL}.log 2>&1; done
echo ВСЁ > pV2_done.flag
