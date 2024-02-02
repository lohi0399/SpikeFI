from time import sleep, time


class CampaignProgress:
    def __init__(self, batches_num: int, rounds_num: int) -> None:
        self.status = 0.
        self.batch = 0
        self.batch_num = batches_num
        self.iter = 0
        self.iter_num = batches_num * rounds_num
        self.fragment = 1. / self.iter_num
        self.start_time = 0.
        self.end_time = 0.

        self._flush_lines_num = 0

    def __str__(self) -> str:
        s = " Batch #\tTotal time\tProgress\n"
        s += f"{self.batch + 1:4d}/{self.batch_num:d}\t"
        if self.start_time:
            s += f"{(time() - self.start_time):6d} sec\t"
        s += f"{self.status * 100.:6.2f} %\t\n"

        self._flush_lines_num = s.count('\n') + 2

        return s

    def has_finished(self) -> bool:
        return self.end_time != 0.

    def get_duration_sec(self) -> float | None:
        if self.has_finished():
            return self.end_time - self.start_time

        return None

    def set_batch(self, b: int) -> None:
        self.batch = b

    def show(self) -> None:
        print('\033[1A\x1b[2K' * self._flush_lines_num)  # Line up, line clear
        print(self)

    def step(self) -> None:
        self.status += self.fragment
        self.iter += 1

    def timer(self) -> None:
        if self.start_time and not self.end_time:
            self.end_time = time()
        else:
            self.start_time = time()
            self.end_time = 0.


def refresh_progress_job(progress: CampaignProgress, secs: float) -> None:
    while progress.iter < progress.iter_num:
        progress.show()
        sleep(secs)
    progress.show()
