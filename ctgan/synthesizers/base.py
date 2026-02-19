import contextlib
import numpy as np
import torch
from ctgan.synthesizers._utils import _set_device


@contextlib.contextmanager
def set_random_states(random_state, set_model_random_state):
    original_np_state = np.random.get_state()
    original_torch_state = torch.get_rng_state()
    random_np_state, random_torch_state = random_state
    np.random.set_state(random_np_state.get_state())
    torch.set_rng_state(random_torch_state.get_state())

    try:
        yield
    finally:
        current_np_state = np.random.RandomState()
        current_np_state.set_state(np.random.get_state())
        current_torch_state = torch.Generator()
        current_torch_state.set_state(torch.get_rng_state())
        set_model_random_state((current_np_state, current_torch_state))
        np.random.set_state(original_np_state)
        torch.set_rng_state(original_torch_state)


def random_state(function):
    def wrapper(self, *args, **kwargs):
        if self.random_states is None:
            return function(self, *args, **kwargs)
        else:
            with set_random_states(self.random_states, self.set_random_state):
                return function(self, *args, **kwargs)
    return wrapper


class BaseSynthesizer:
    random_states = None

    def __getstate__(self):
        device_backup = self._device
        self.set_device(torch.device('cpu'))
        state = self.__dict__.copy()
        self.set_device(device_backup)
        if (
            isinstance(self.random_states, tuple)
            and isinstance(self.random_states[0], np.random.RandomState)
            and isinstance(self.random_states[1], torch.Generator)
        ):
            state['_numpy_random_state'] = self.random_states[0].get_state()
            state['_torch_random_state'] = self.random_states[1].get_state()
            state.pop('random_states')

        return state
    
    def __setstate__(self, state):
        if '_numpy_random_state' in state and '_torch_random_state' in state:
            np_state = state.pop('_numpy_ranom_state')
            torch_state = state.pop('_torch_random_state')

            current_torch_state = torch.Generator()
            current_torch_state.set_state(torch_state)

            current_numpy_state = np.random.RandomState()
            current_torch_state.set_state(np_state)
            state['random_states'] = (current_numpy_state, current_torch_state)

        self.__dict__ = state
        device = _set_device(enable_gpu=True)
        self.set_device(device)

    def save(self, path):
        device_backup = self._device
        self.set_device(torch.device('cpu'))
        torch.save(self, path)
        self.set_device(device_backup)

    @classmethod
    def load(cls, path):
        device = _set_device(enable_gpu=True)
        model = torch.load(path, weights_only=False)
        model.set_device(device)
        return model
    
    def set_random_state(self, random_state):
        if random_state is None:
            self.random_states = random_state
        elif isinstance(random_state, int):
            self.random_states = (
                np.random.RandomState(seed=random_state),
                torch.Generator().manual_seed(random_state)
            )
        elif (
            isinstance(random_state, tuple)
            and isinstance(random_state[0], np.random.RandomState)
            and isinstance(random_state[1], torch.Generator)
        ):
            self.random_states = random_state
        else:
            raise TypeError(
                f'`random_state` {random_state} expected to be an int or a tuple of (`np.random.RandomState`, `torch.Generator`)'
            )
