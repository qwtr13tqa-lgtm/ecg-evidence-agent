"""Nonblocking process ownership lock, Windows and POSIX. Keep file until close."""
import errno
import os


def acquire_history_lock(path):
    handle=open(path,'a+b')
    try:
        if os.name=='nt':
            import msvcrt
            # Lock the first byte, always from offset zero. Never truncate or unlink.
            handle.seek(0,os.SEEK_END)
            if handle.tell()==0:
                handle.write(b'\0');handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return handle
    except OSError as exc:
        handle.close()
        if exc.errno in (errno.EACCES,errno.EAGAIN,errno.EDEADLK):
            raise RuntimeError('另一个服务正在使用历史数据库，请停止旧服务后重启。') from exc
        raise
    except BaseException:
        handle.close()
        raise
