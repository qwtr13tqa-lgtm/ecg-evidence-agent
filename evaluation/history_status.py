"""Inspect counts and optionally back up history; no inference or gateway."""
import argparse
from pathlib import Path
from datetime import datetime,timezone
from src.analysis.history import HistoryRepository


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backup',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[1];path=root/'local_history/history.sqlite3'
    if not path.exists():print('No history database yet. Save one analysis from the new UI.');return
    repo=HistoryRepository(path,recover=False)
    with repo.connect() as db:
        for table in ('analyses','conversations','turns'):
            print(table,db.execute('SELECT count(*) FROM '+table).fetchone()[0])
    print('database_bytes',path.stat().st_size)
    if a.backup:
        name='history-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.sqlite3'
        print('Backup:',repo.backup(root/'local_history/backups'/name))

if __name__=='__main__':main()
