import numpy as np

from collections import deque


class FrameHelper:
    """ classe che si occupa di creare uno stack lungo stack_size di frame """
    def __init__(self, stack_size):
        self.stack_size = stack_size

    def stack_frames(self, stacked_frames, frame, is_new_episode):
        """ crea uno stack di frame lungo stack_size """
        if is_new_episode:
            # svuota la nostra coda di frame
            stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(self.stack_size)], maxlen=4)

            # siccome ci troviamo in un nuovo episodio allora copiamo lostesso frame 4 volte
            for _ in range(self.stack_size):
                stacked_frames.append(frame)

            # creo lo stack di frames
            stacked_state = np.stack(stacked_frames, axis=2)

        else:
            # appendo il frame alla coda e automaticamente elimina il frame più vecchio (FIFO)
            stacked_frames.append(frame)

            # creo lo stck di frames
            stacked_state = np.stack(stacked_frames, axis=2)

        return stacked_state, stacked_frames

