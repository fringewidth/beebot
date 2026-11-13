"""A read-only seekable file over HTTP range requests (via curl), so a large archive (7z) can be read without
storing it. curl is used because the bucket answered python's urllib with 403 at times."""

import io
import subprocess


class HttpFile(io.RawIOBase):
    def __init__(self, url, chunk=8 << 20):
        self.url = url
        self.chunk = chunk
        head = subprocess.run(["curl", "-sIL", "-m", "60", url], capture_output=True, text=True, check=True).stdout
        sizes = [int(line.split(":", 1)[1]) for line in head.splitlines() if line.lower().startswith("content-length:")]
        if not sizes:
            raise OSError(f"no content-length for {url}")
        self.size = sizes[-1]
        self.pos = 0
        self.cache_start = 0
        self.cache = b""

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def _fetch(self, start):
        end = min(start + self.chunk, self.size) - 1
        for attempt in range(5):
            run = subprocess.run(["curl", "-sL", "-m", "300", "-r", f"{start}-{end}", self.url], capture_output=True)
            if run.returncode == 0 and len(run.stdout) == end - start + 1:
                return run.stdout
        raise OSError(f"range {start}-{end} failed")

    def readinto(self, buffer):
        want = len(buffer)
        if self.pos >= self.size or want == 0:
            return 0
        if not (self.cache_start <= self.pos < self.cache_start + len(self.cache)):
            self.cache_start = self.pos
            self.cache = self._fetch(self.pos)
        offset = self.pos - self.cache_start
        data = self.cache[offset : offset + want]
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)
