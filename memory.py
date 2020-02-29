import numpy as np

from costants import POST_PROCESS_IMAGE_SIZE, BATCH_SIZE, NUM_FRAMES


class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._actions = np.zeros(max_memory, dtype=np.int32)
        self._rewards = np.zeros(max_memory, dtype=np.float32)
        self._frames = np.zeros((POST_PROCESS_IMAGE_SIZE[0], POST_PROCESS_IMAGE_SIZE[1], max_memory), dtype=np.float32)
        self._done = np.zeros(max_memory, dtype=np.bool)
        self._i = 0

    def add_sample(self, frame, action, reward, done):
        self._actions[self._i] = action
        self._rewards[self._i] = reward
        self._frames[:, :, self._i] = frame[:, :, 0]
        self._done[self._i] = done
        if self._i % (self._max_memory - 1) == 0 and self._i != 0:
            self._i = BATCH_SIZE + NUM_FRAMES + 1
        else:
            self._i += 1

    def get_samples(self):
        if self._i < BATCH_SIZE + NUM_FRAMES + 1:
            raise ValueError("Non ci sono abbastanza dati in memoria per estrarne uno.")
        else:
            rand_idxs = np.random.randint(NUM_FRAMES + 1, self._i, size=BATCH_SIZE)
            states = np.zeros((BATCH_SIZE, POST_PROCESS_IMAGE_SIZE[0], POST_PROCESS_IMAGE_SIZE[1], NUM_FRAMES),
                              dtype=np.float32)
            next_states = np.zeros((BATCH_SIZE, POST_PROCESS_IMAGE_SIZE[0], POST_PROCESS_IMAGE_SIZE[1], NUM_FRAMES),
                                   dtype=np.float32)
            for i, idx in enumerate(rand_idxs):
                states[i] = self._frames[:, :, idx - 1 - NUM_FRAMES:idx - 1]
                next_states[i] = self._frames[:, :, idx - NUM_FRAMES:idx]
            return states, self._actions[rand_idxs], self._rewards[rand_idxs], next_states, self._done[rand_idxs]

    def restore_memory(self, actions, rewards, frames, done, i):
        self._actions = actions
        self._rewards = rewards
        self._frames = frames
        self._done = done
        self._i = i

    def get_memory(self):
        return self._actions, self._rewards, self._frames, self._done, self._i