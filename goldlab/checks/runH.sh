cd /home/claude/code/goldlab_checks
python3 -u onebody.py 16 18 20 24 > h_onebody.log 2>&1
for LL in 16 18 20; do L=$LL M1=1 M2=5 C=exact MEAS=strip CAP=300 python3 -u pair_c2.py > h_strip_L$LL.log 2>&1; done
L=16 M1=1 M2=5 F=0.08 CAP=150 python3 -u cov.py > h_cov_L16.log 2>&1
echo ВСЁ > h_done.flag
