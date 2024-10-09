import re
from time import sleep, time


class CampaignProgress:
    def __init__(self, batches_num: int, rounds_num: int, epochs: int = None) -> None:
        self.is_training = bool(epochs)
        self.loss = 0.
        self.accu = 0.
        self.status = 0.
        self.batch = 0
        self.batch_num = batches_num
        self.epoch = 0
        self.epochs = epochs
        self.iter = 0
        self.iter_num = batches_num * rounds_num * (epochs or 1)
        self.fragment = 1. / self.iter_num
        self.start_time = 0.
        self.end_time = 0.

        self._flush_lines_num = 0

    def __str__(self) -> str:
        s = "|  Batch #  | Total time | Progress |\n"

        if self.is_training:
            e = self.epoch + 1 if self.epoch + 1 <= self.epochs else self.epochs

            s = "|  Loss   | Accuracy | Epoch # " + s
            border = re.sub(r'[^+\n]', '-', s.replace('|', '+'))
            s = border + s

            s += f"| {self.loss:<7.3f} | {self.accu * 100.:6.3f} % | {e:3d}/{self.epochs:<3d}"
        else:
            border = re.sub(r'[^+\n]', '-', s.replace('|', '+'))
            s = border + s

        b = self.batch + 1 if self.batch + 1 <= self.batch_num else self.batch_num

        s += f" | {b:4d}/{self.batch_num:<4d} | "
        if self.start_time:
            s += f"{(time() - self.start_time):6.0f} sec | "
        s += f"{self.status * 100.:6.2f} % |\n"
        s += border

        self._flush_lines_num = s.count('\n') + 2

        return s

    def has_finished(self) -> bool:
        return self.end_time != 0.

    def get_duration_sec(self) -> float | None:
        if self.has_finished():
            return self.end_time - self.start_time

        return None

    def reset_epoch(self) -> None:
        self.epoch = 0
        self.batch = 0

    def show(self) -> None:
        print('\033[1A\x1b[2K' * self._flush_lines_num)  # Line up, line clear
        print(self)

    def set_train(self, loss: float, accu: float) -> None:
        if loss:
            self.loss = loss
        if accu:
            self.accu = accu

    def step(self) -> None:
        self.status += self.fragment
        self.iter += 1

    def step_batch(self) -> None:
        self.batch += 1

    def step_epoch(self) -> None:
        self.epoch += 1
        if self.epoch < self.epochs:
            self.batch = 0

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
