#!/bin/bash
# Backfill match stats 2015-2026 with force (re-scrape over broken cache)
cd /root/NRL-Predict
python3 -c "
from scraping.nrl_match_stats import backfill_all_stats
print('Starting backfill 2015-2026 (force=True)...')
df = backfill_all_stats(start_year=2015, end_year=2026, force=True, delay=0.4)
print(f'Done: {len(df)} total match stat rows')
" > /tmp/backfill_match_stats.log 2>&1
