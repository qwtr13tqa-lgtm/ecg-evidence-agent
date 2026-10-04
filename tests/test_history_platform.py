"""Native subprocess exclusion, mock Windows branch, and init failure cleanup."""
import errno
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from src.analysis.history import HistoryRepository
from src.analysis.history_lock import acquire_history_lock

class HistoryPlatformTests(unittest.TestCase):
    def test_native_process_exclusion_and_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            db=Path(tmp)/'history.sqlite'
            repo=HistoryRepository(db,exclusive=True)
            script="from src.analysis.history import HistoryRepository; import sys; r=HistoryRepository(sys.argv[1],exclusive=True); r.close()"
            first=subprocess.run([sys.executable,'-c',script,str(db)],capture_output=True)
            self.assertNotEqual(first.returncode,0)
            repo.close()
            second=subprocess.run([sys.executable,'-c',script,str(db)],capture_output=True)
            self.assertEqual(second.returncode,0,second.stderr)
    def test_windows_locks_first_byte(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=str(Path(tmp)/'file.lock');calls=[]
            def locking(fd,mode,count):
                calls.append((os.lseek(fd,0,os.SEEK_CUR),mode,count))
            with patch('src.analysis.history_lock.os.name','nt'),patch.dict(sys.modules,{'msvcrt':SimpleNamespace(LK_NBLCK=2,locking=locking)}):
                h=acquire_history_lock(path);h.close()
            self.assertEqual(calls,[(0,2,1)])
    def test_windows_contention_closes_handle(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=str(Path(tmp)/'file.lock')
            def locking(*args):raise OSError(errno.EACCES,'busy')
            with patch('src.analysis.history_lock.os.name','nt'),patch.dict(sys.modules,{'msvcrt':SimpleNamespace(LK_NBLCK=2,locking=locking)}):
                with self.assertRaises(RuntimeError):acquire_history_lock(path)
            h=acquire_history_lock(path);h.close()
    def test_initialization_failure_releases_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            db=Path(tmp)/'history.sqlite';db.write_bytes(b'not a database')
            with self.assertRaises(Exception):HistoryRepository(db,exclusive=True)
            h=acquire_history_lock(str(db)+'.lock');h.close()

if __name__=='__main__':unittest.main()
